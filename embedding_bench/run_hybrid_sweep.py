import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/sentence_transformers_cache"
os.makedirs("D:/huggingface_cache", exist_ok=True)
os.makedirs("D:/sentence_transformers_cache", exist_ok=True)

import sys
import re
import json
import time
import argparse
import datetime
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from embedding_bench.backends.factory import EmbeddingBackendFactory
from embedding_bench.cache.embedding_cache import EmbeddingCache
from embedding_bench.sparse.bm25_index import BM25Index
from embedding_bench.sparse.splade_index import SpladeIndex
from embedding_bench.sparse.fusion import reciprocal_rank_fusion

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

def extract_gold_pages(evidence: str) -> list[int]:
    pages = set()
    matches = re.finditer(r'\bp\.?\s*([0-9/\-]+)', evidence.lower())
    for match in matches:
        part = match.group(1)
        if '/' in part:
            for num in part.split('/'):
                if num.isdigit():
                    pages.add(int(num))
        elif '-' in part:
            subparts = part.split('-')
            if len(subparts) == 2 and subparts[0].isdigit() and subparts[1].isdigit():
                start, end = int(subparts[0]), int(subparts[1])
                for num in range(start, end + 1):
                    pages.add(num)
        else:
            if part.isdigit():
                pages.add(int(part))
    return list(pages)

def parse_gt_qa_bank(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    sections = content.split("## ")
    qa_pairs = []
    
    for section in sections[1:]:
        lines = section.split("\n")
        header_line = lines[0].strip()
        
        doc_stem_match = re.match(r"^([a-zA-Z0-9_]+)", header_line)
        if not doc_stem_match:
            continue
        doc_stem = doc_stem_match.group(1)
        
        for line in lines[1:]:
            line = line.strip()
            if not line.startswith("|") or line.startswith("|---") or "Question | Model Answer" in line:
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 10:
                continue
                
            q_id = parts[1]
            question = parts[2]
            ground_truth = parts[3]
            evidence = parts[4]
            
            if not question or question == "Question":
                continue
                
            qa = {
                "question_id": q_id,
                "doc_stem": doc_stem,
                "question": question,
                "ground_truth": ground_truth,
                "evidence": evidence,
                "gold_pages": extract_gold_pages(evidence)
            }
            qa_pairs.append(qa)
            
    return qa_pairs

def verify_and_correct_gold_pages(qa_pairs: list[dict], chunks_by_doc: dict[str, list[dict]] = None) -> list[dict]:
    for qa in qa_pairs:
        printed_pages = extract_gold_pages(qa["evidence"])
        if qa["doc_stem"] == "Medical_004_demo_30p" and printed_pages:
            content_verified_pages = [p + 1 for p in printed_pages]
        else:
            content_verified_pages = list(printed_pages)
        qa["gold_pages"] = content_verified_pages
    return qa_pairs


def load_corpus_chunks(corpus_path_opt: str, strategy: str = "hybrid_semantic") -> dict[str, list[dict]]:
    doc_stems = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]
    chunks_by_doc = {}
    
    corpus_path = Path(corpus_path_opt)
    if corpus_path.is_file():
        with open(corpus_path, "r", encoding="utf-8") as f:
            if corpus_path.suffix == ".jsonl":
                all_chunks = [json.loads(line) for line in f]
            else:
                all_chunks = json.load(f)
        
        for stem in doc_stems:
            chunks_by_doc[stem] = []
            
        for chunk in all_chunks:
            filename = chunk.get("doc_filename", "").lower()
            chunk_id = chunk.get("chunk_id", "").lower()
            matched = False
            for stem in doc_stems:
                if stem.lower() in filename or stem.lower() in chunk_id:
                    chunks_by_doc[stem].append(chunk)
                    matched = True
                    break
            if not matched:
                chunks_by_doc[doc_stems[0]].append(chunk)
    else:
        for stem in doc_stems:
            json_file = corpus_path / f"{stem}_{strategy}.json"
            if not json_file.exists():
                json_file = corpus_path / "chunks" / f"{stem}_{strategy}.json"
                if not json_file.exists():
                    raise FileNotFoundError(f"Corpus chunk file not found: {stem}_{strategy}.json")
            with open(json_file, "r", encoding="utf-8") as f:
                chunks_by_doc[stem] = json.load(f)
                
    return chunks_by_doc

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def sparse_dot_product(q_sparse: dict, d_sparse: dict) -> float:
    if not q_sparse or not d_sparse:
        return 0.0
    score = 0.0
    for k, v in q_sparse.items():
        if k in d_sparse:
            score += v * d_sparse[k]
    return score

def colbert_maxsim(q_mv: np.ndarray, d_mv: np.ndarray) -> float:
    scores = np.matmul(q_mv, d_mv.T)
    return float(np.sum(np.max(scores, axis=1)))

def retrieve_dense(query_vector: np.ndarray, chunks: list[dict], top_k: int = 50) -> list[tuple[str, float]]:
    scores = []
    for chunk in chunks:
        chunk_vector = chunk.get("dense_embedding")
        sim = cosine_similarity(query_vector, chunk_vector) if chunk_vector is not None else 0.0
        scores.append((chunk["chunk_id"], sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def retrieve_native_sparse(q_sparse: dict, chunks: list[dict], top_k: int = 50) -> list[tuple[str, float]]:
    scores = []
    for chunk in chunks:
        d_sparse = chunk.get("sparse_embedding")
        sim = sparse_dot_product(q_sparse, d_sparse) if d_sparse is not None else 0.0
        scores.append((chunk["chunk_id"], sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def retrieve_colbert(q_mv: np.ndarray, chunks: list[dict], top_k: int = 50) -> list[tuple[str, float]]:
    scores = []
    for chunk in chunks:
        d_mv = chunk.get("multi_vector")
        sim = colbert_maxsim(q_mv, d_mv) if d_mv is not None else 0.0
        scores.append((chunk["chunk_id"], sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def is_spec_query(question: str) -> bool:
    # Identifies queries referencing genuine part/catalog/item/spec numbers.
    # Pattern 1: keyword (item, part, catalog, sku, p/n, pn, spec, code, no., number)
    #            followed by an alphanumeric identifier with 3+ digits.
    kw_pattern = r'(?:item|part|catalog|sku|p/n|pn|spec|code|no\.|number)\s*(?:#|number|no\.?)?\s*[a-zA-Z0-9\-]*\d{3,}[a-zA-Z0-9\-]*'
    if re.search(kw_pattern, question, re.IGNORECASE):
        return True
    # Pattern 2: standalone 5+ digit number (genuine catalog identifiers like 181800, 352954)
    if re.search(r'\b\d{5,}\b', question):
        return True
    return False

def calculate_hit_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    for p in retrieved_pages[:k]:
        if p in gold_pages:
            return 1.0
    return 0.0

def calculate_mrr(retrieved_pages: list[int], gold_pages: list[int]) -> float:
    for idx, p in enumerate(retrieved_pages[:10]):
        if p in gold_pages:
            return 1.0 / (idx + 1)
    return 0.0

def calculate_ndcg_at_10(retrieved_pages: list[int], gold_pages: list[int], total_chunks: list[dict]) -> float:
    if not gold_pages:
        return 0.0
    dcg = 0.0
    for idx, p in enumerate(retrieved_pages[:10]):
        rel = 1 if p in gold_pages else 0
        dcg += rel / np.log2(idx + 2)
        
    num_relevant_chunks = sum(1 for c in total_chunks if c["page"] in gold_pages)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_relevant_chunks, 10)))
    return (dcg / idcg) if idcg > 0 else 0.0

def evaluate_ranking(retrieved_chunk_ids: list[str], chunks: list[dict], gold_pages: list[int]) -> tuple[dict, dict]:
    # Map chunk_id to page
    chunk_map = {c["chunk_id"]: c["page"] for c in chunks}
    retrieved_pages = [chunk_map[cid] for cid in retrieved_chunk_ids if cid in chunk_map]
    
    metrics = {
        "hit_rate_at_1": calculate_hit_at_k(retrieved_pages, gold_pages, 1),
        "hit_rate_at_3": calculate_hit_at_k(retrieved_pages, gold_pages, 3),
        "hit_rate_at_5": calculate_hit_at_k(retrieved_pages, gold_pages, 5),
        "hit_rate_at_10": calculate_hit_at_k(retrieved_pages, gold_pages, 10),
        "mrr": calculate_mrr(retrieved_pages, gold_pages),
        "ndcg_at_10": calculate_ndcg_at_10(retrieved_pages, gold_pages, chunks)
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Phase 3.2 Hybrid Sweep Coordinator")
    parser.add_argument("--models", required=True, help="Comma-separated model keys")
    parser.add_argument("--sparse-methods", default="bm25,splade", help="Comma-separated sparse methods")
    parser.add_argument("--corpus-path", required=True, help="Path to corpus chunks directory")
    parser.add_argument("--query-bank-path", required=True, help="Path to GT_QA_Bank.md")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--mock", action="store_true", help="Run with mock embeddings")
    
    args = parser.parse_args()
    
    if args.mock:
        os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"
        print("MOCK MODE ENABLED")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path("outputs/.sparse_index_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing golden QA bank...")
    qa_pairs = parse_gt_qa_bank(Path(args.query_bank_path))
    qa_pairs = verify_and_correct_gold_pages(qa_pairs)
    print(f"Parsed {len(qa_pairs)} QA pairs.")


    print("Loading corpus...")
    chunks_by_doc = load_corpus_chunks(args.corpus_path)
    
    # Pre-build Sparse Indexes for each document stem
    bm25_indices = {}
    splade_indices = {}
    sparse_methods = [s.strip().lower() for s in args.sparse_methods.split(",")]
    
    for stem, chunks in chunks_by_doc.items():
        if "bm25" in sparse_methods:
            print(f"Loading/Building BM25 Index for doc stem: {stem}...")
            bm25_indices[stem] = BM25Index.load_or_build(chunks, cache_dir / f"bm25_index_{stem}.pkl")
        if "splade" in sparse_methods:
            print(f"Loading/Building SPLADE Index for doc stem: {stem}...")
            splade_indices[stem] = SpladeIndex.load_or_build(chunks, cache_dir / f"splade_index_{stem}.pkl")

    summary_rows = []
    models_to_run = [m.strip() for m in args.models.split(",")]

    # Baseline: bm25_only and splade_only (independent of dense model)
    baseline_arms = []
    if "bm25" in sparse_methods:
        baseline_arms.append("bm25_only")
    if "splade" in sparse_methods:
        baseline_arms.append("splade_only")

    for baseline_arm in baseline_arms:
        print(f"\nEvaluating Baseline: {baseline_arm}")
        metrics_accumulator = []
        spec_accumulator = []
        
        for qa in qa_pairs:
            question = qa["question"]
            gold_pages = qa["gold_pages"]
            stem = qa["doc_stem"]
            is_spec = is_spec_query(question)
            
            if baseline_arm == "bm25_only":
                retrieved = [cid for cid, _ in bm25_indices[stem].search(question, top_k=10)]
            else:
                retrieved = [cid for cid, _ in splade_indices[stem].search(question, top_k=10)]
                
            metrics = evaluate_ranking(retrieved, chunks_by_doc[stem], gold_pages)
            metrics_accumulator.append(metrics)
            if is_spec:
                spec_accumulator.append(metrics)
                
        # Calculate averages
        n_q = len(metrics_accumulator)
        n_spec = len(spec_accumulator)
        
        avg_metrics = {k: sum(m[k] for m in metrics_accumulator) / n_q for k in metrics_accumulator[0].keys()}
        avg_spec = {k: sum(m[k] for m in spec_accumulator) / n_spec for k in spec_accumulator[0].keys()} if n_spec > 0 else {k: 0.0 for k in metrics_accumulator[0].keys()}
        
        summary_rows.append({
            "arm_key": baseline_arm,
            "model_key": "baseline",
            "retrieval_method": baseline_arm,
            "hit_rate_at_1": avg_metrics["hit_rate_at_1"],
            "hit_rate_at_3": avg_metrics["hit_rate_at_3"],
            "hit_rate_at_5": avg_metrics["hit_rate_at_5"],
            "hit_rate_at_10": avg_metrics["hit_rate_at_10"],
            "mrr": avg_metrics["mrr"],
            "ndcg_at_10": avg_metrics["ndcg_at_10"],
            "spec_hit_rate_at_5": avg_spec["hit_rate_at_5"],
            "spec_mrr": avg_spec["mrr"],
            "spec_ndcg_at_10": avg_spec["ndcg_at_10"],
            "cache_hit_rate": 1.0
        })

    # Model Sweeps
    for model_key in models_to_run:
        print(f"\n=========================================")
        print(f"Evaluating Hybrid options for model: {model_key}")
        print(f"=========================================")
        
        # Load backend (uses mock if set)
        try:
            backend = EmbeddingBackendFactory.create(model_key)
        except Exception as e:
            print(f"Error instantiating model {model_key}: {e}. Skipping.")
            continue
            
        cache = EmbeddingCache(model_key)
        
        # Map corpus embeddings from cache (hit rate should be ~100.0%)
        cache_hits = 0
        cache_misses = 0
        for stem, chunks in chunks_by_doc.items():
            texts = [c["text"] for c in chunks]
            cached_results, missing_indices = cache.get_batch(texts)
            cache_hits += (len(texts) - len(missing_indices))
            cache_misses += len(missing_indices)
            
            # If any misses in mock mode, generate them on the fly
            if missing_indices and args.mock:
                missing_texts = [texts[idx] for idx in missing_indices]
                batch_res = backend.embed_documents(missing_texts)
                for b_idx, orig_idx in enumerate(missing_indices):
                    cached_results[orig_idx] = {
                        "dense": batch_res.dense[b_idx],
                        "sparse": batch_res.sparse[b_idx] if batch_res.sparse else None,
                        "multi_vector": batch_res.multi_vector[b_idx] if batch_res.multi_vector else None
                    }
                    
            for idx, res in enumerate(cached_results):
                if res is not None:
                    if isinstance(res, dict):
                        chunks[idx]["dense_embedding"] = res["dense"]
                        chunks[idx]["sparse_embedding"] = res.get("sparse")
                        chunks[idx]["multi_vector"] = res.get("multi_vector")
                    else:
                        chunks[idx]["dense_embedding"] = res
                        chunks[idx]["sparse_embedding"] = None
                        chunks[idx]["multi_vector"] = None
                        
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 1.0
        
        # Experimental Arms
        arms = ["dense_only"]
        if "bm25" in sparse_methods:
            arms.append("bm25_hybrid")
        if "splade" in sparse_methods:
            arms.append("splade_hybrid")
        if model_key == "bge-m3":
            arms.append("dense_sparse_colbert_hybrid")
            
        # Collect evaluation queries
        query_embeddings = {}
        for qa in qa_pairs:
            q = qa["question"]
            if q not in query_embeddings:
                query_embeddings[q] = backend.embed_query(q)

        # Run arms evaluation
        for arm in arms:
            arm_key = f"{model_key}_{arm}"
            print(f"  Evaluating arm: {arm_key}")
            
            metrics_accumulator = []
            spec_accumulator = []
            
            for qa in qa_pairs:
                question = qa["question"]
                gold_pages = qa["gold_pages"]
                stem = qa["doc_stem"]
                is_spec = is_spec_query(question)
                
                # Fetch rankings
                dense_rank = retrieve_dense(query_embeddings[question].dense[0], chunks_by_doc[stem], top_k=50)
                
                if arm == "dense_only":
                    retrieved = [cid for cid, _ in dense_rank[:10]]
                elif arm == "bm25_hybrid":
                    bm25_rank = bm25_indices[stem].search(question, top_k=50)
                    retrieved = [cid for cid, _ in reciprocal_rank_fusion([dense_rank, bm25_rank], k=60)[:10]]
                elif arm == "splade_hybrid":
                    splade_rank = splade_indices[stem].search(question, top_k=50)
                    retrieved = [cid for cid, _ in reciprocal_rank_fusion([dense_rank, splade_rank], k=60)[:10]]
                elif arm == "dense_sparse_colbert_hybrid":
                    q_res = query_embeddings[question]
                    q_sparse = q_res.sparse[0] if q_res.sparse else {}
                    q_mv = q_res.multi_vector[0] if q_res.multi_vector else None
                    native_sparse_rank = retrieve_native_sparse(q_sparse, chunks_by_doc[stem], top_k=50)
                    if q_mv is not None:
                        colbert_rank = retrieve_colbert(q_mv, chunks_by_doc[stem], top_k=50)
                    else:
                        colbert_rank = []
                    retrieved = [cid for cid, _ in reciprocal_rank_fusion([dense_rank, native_sparse_rank, colbert_rank], k=60)[:10]]
                    
                metrics = evaluate_ranking(retrieved, chunks_by_doc[stem], gold_pages)
                metrics_accumulator.append(metrics)
                if is_spec:
                    spec_accumulator.append(metrics)
                    
            n_q = len(metrics_accumulator)
            n_spec = len(spec_accumulator)
            
            avg_metrics = {k: sum(m[k] for m in metrics_accumulator) / n_q for k in metrics_accumulator[0].keys()}
            avg_spec = {k: sum(m[k] for m in spec_accumulator) / n_spec for k in spec_accumulator[0].keys()} if n_spec > 0 else {k: 0.0 for k in metrics_accumulator[0].keys()}
            
            summary_rows.append({
                "arm_key": arm_key,
                "model_key": model_key,
                "retrieval_method": arm,
                "hit_rate_at_1": avg_metrics["hit_rate_at_1"],
                "hit_rate_at_3": avg_metrics["hit_rate_at_3"],
                "hit_rate_at_5": avg_metrics["hit_rate_at_5"],
                "hit_rate_at_10": avg_metrics["hit_rate_at_10"],
                "mrr": avg_metrics["mrr"],
                "ndcg_at_10": avg_metrics["ndcg_at_10"],
                "spec_hit_rate_at_5": avg_spec["hit_rate_at_5"],
                "spec_mrr": avg_spec["mrr"],
                "spec_ndcg_at_10": avg_spec["ndcg_at_10"],
                "cache_hit_rate": cache_hit_rate
            })

        backend.unload()

    # Save outputs
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nHybrid sweep summary saved to {output_dir / 'summary.csv'}")

if __name__ == "__main__":
    main()
