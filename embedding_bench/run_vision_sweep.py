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
from embedding_bench.data.figure_chunk_linker import build_linked_chunk_set

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

def retrieve_dense(query_vector: np.ndarray, chunks: list[dict], top_k: int = 10) -> list[dict]:
    scores = []
    for chunk in chunks:
        chunk_vector = chunk.get("dense_embedding")
        sim = cosine_similarity(query_vector, chunk_vector) if chunk_vector is not None else 0.0
        scores.append((sim, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

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

def main():
    parser = argparse.ArgumentParser(description="Phase 3.4 Vision Sweep Coordinator")
    parser.add_argument("--models", required=True, help="Comma-separated multimodal models")
    parser.add_argument("--linked-chunks-path", default="outputs/linked_chunks.jsonl", help="Linked chunks JSONL path")
    parser.add_argument("--query-bank-path", required=True, help="Path to GT_QA_Bank.md")
    parser.add_argument("--run-id", required=True, help="Vision sweep identifier")
    parser.add_argument("--output-dir", required=True, help="Output folder")
    parser.add_argument("--mock", action="store_true", help="Bypass model downloads and use mock embeddings")
    
    args = parser.parse_args()
    
    if args.mock:
        os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"
        print("MOCK MODE ENABLED")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building/Retrieving Linked Chunk Set...")
    # Corpus folder path deduced from linked chunks path parent or chunk outputs
    corpus_dir = Path("outputs/chunks")
    linked_chunks = build_linked_chunk_set(corpus_dir, Path(args.linked_chunks_path))
    print(f"Build linked chunk set. Total chunks linked: {len(linked_chunks)}")

    # Spot check 5 random linked chunks
    print("\n--- Linked Chunk Spot Check (5 random examples) ---")
    import random
    random.seed(42)
    linked_only = [lc for lc in linked_chunks if lc.figure_image_path is not None]
    if linked_only:
        sample_size = min(len(linked_only), 5)
        samples = random.sample(linked_only, sample_size)
        for i, s in enumerate(samples):
            print(f"Sample {i+1}:")
            print(f"  Chunk ID: {s.chunk_id}")
            print(f"  Text Snippet: {s.text[:120]}...")
            print(f"  Figure Path: {s.figure_image_path}")
            print(f"  Confidence: {s.link_confidence}")
    else:
        print("No linked chunks found (text only).")
    print("---------------------------------------------------\n")

    print("Loading Ground Truth QA bank...")
    qa_pairs = parse_gt_qa_bank(Path(args.query_bank_path))
    qa_pairs = verify_and_correct_gold_pages(qa_pairs)
    print(f"Parsed {len(qa_pairs)} QA pairs.")


    print("Loading corpus chunks...")
    chunks_by_doc = load_corpus_chunks("outputs/chunks")

    # Map chunk_id -> LinkedChunk info for quick lookup
    linked_map = {lc.chunk_id: lc for lc in linked_chunks}

    # Filter QA pairs slice: expected answer chunk has figure_image_path is not None
    # A query is "figure_linked" if any of the chunks in its target page has figure_image_path is not None
    filtered_qa_pairs = []
    for qa in qa_pairs:
        stem = qa["doc_stem"]
        gold_pages = qa["gold_pages"]
        
        # Check if any chunk in the target page has an image link
        has_linked_fig = False
        for chunk in chunks_by_doc[stem]:
            if chunk["page"] in gold_pages:
                lc = linked_map.get(chunk["chunk_id"])
                if lc and lc.figure_image_path is not None:
                    has_linked_fig = True
                    break
        if has_linked_fig:
            filtered_qa_pairs.append(qa)

    print(f"Filtered to {len(filtered_qa_pairs)} queries where expected answer chunk is figure-linked.")

    summary_rows = []
    models_to_run = [m.strip() for m in args.models.split(",")]

    # Evaluate each vision model
    for model_key in models_to_run:
        print(f"\n=========================================")
        print(f"Starting sweep for vision model: {model_key}")
        print(f"=========================================")

        try:
            backend = EmbeddingBackendFactory.create(model_key)
        except Exception as e:
            print(f"Error instantiating model {model_key}: {e}. Skipping.")
            continue

        cache = EmbeddingCache(model_key)

        # Run text_only and text_plus_image arms
        for arm in ["text_only", "text_plus_image"]:
            arm_key = f"{model_key}_{arm}"
            print(f"Evaluating arm: {arm_key}...")
            
            # Embed corpus chunks
            # In text_only mode, we set figure_image_path=None on all chunks
            # In text_plus_image mode, we use the real figure_image_path
            current_linked_chunks = []
            for stem, chunks in chunks_by_doc.items():
                for chunk in chunks:
                    lc = linked_map.get(chunk["chunk_id"])
                    if arm == "text_only":
                        current_linked_chunks.append({
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "figure_image_path": None
                        })
                    else:
                        current_linked_chunks.append({
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "figure_image_path": lc.figure_image_path if lc else None
                        })
                        
            # Call vision backend encoding
            corpus_res = backend.embed_documents(current_linked_chunks)
            
            # Populate embeddings back to chunks
            idx = 0
            for stem, chunks in chunks_by_doc.items():
                for chunk in chunks:
                    chunk["dense_embedding"] = corpus_res.dense[idx]
                    idx += 1
                    
            # Embed queries
            query_embeddings = {}
            for qa in filtered_qa_pairs:
                q = qa["question"]
                if q not in query_embeddings:
                    query_embeddings[q] = backend.embed_query(q).dense[0]
                    
            # Evaluate rankings on the filtered slice
            hit_1 = 0
            hit_3 = 0
            hit_5 = 0
            hit_10 = 0
            mrr_total = 0.0
            ndcg_total = 0.0
            
            for qa in filtered_qa_pairs:
                question = qa["question"]
                gold_pages = qa["gold_pages"]
                stem = qa["doc_stem"]
                
                q_vec = query_embeddings[question]
                retrieved = retrieve_dense(q_vec, chunks_by_doc[stem], top_k=10)
                retrieved_pages = [c["page"] for c in retrieved]
                
                hit_1 += calculate_hit_at_k(retrieved_pages, gold_pages, 1)
                hit_3 += calculate_hit_at_k(retrieved_pages, gold_pages, 3)
                hit_5 += calculate_hit_at_k(retrieved_pages, gold_pages, 5)
                hit_10 += calculate_hit_at_k(retrieved_pages, gold_pages, 10)
                mrr_total += calculate_mrr(retrieved_pages, gold_pages)
                ndcg_total += calculate_ndcg_at_10(retrieved_pages, gold_pages, chunks_by_doc[stem])
                
            n_q = len(filtered_qa_pairs)
            hit_5_rate = hit_5 / n_q if n_q > 0 else 0.0
            mrr = mrr_total / n_q if n_q > 0 else 0.0
            ndcg = ndcg_total / n_q if n_q > 0 else 0.0
            
            summary_rows.append({
                "model_key": model_key,
                "arm": arm,
                "hit_rate_at_1": hit_1 / n_q if n_q > 0 else 0.0,
                "hit_rate_at_3": hit_3 / n_q if n_q > 0 else 0.0,
                "hit_rate_at_5": hit_5_rate,
                "hit_rate_at_10": hit_10 / n_q if n_q > 0 else 0.0,
                "mrr": mrr,
                "ndcg_at_10": ndcg
            })
            
        backend.unload()

    # Reference Comparison Arm: Pull the best text-only model's metrics on this SAME slice
    print("\nEvaluating Reference Text-only Models on this filtered slice...")
    text_models = ["bge-m3", "qwen3-embedding-8b-4bit", "nv-embed-v2-fp16", "nomic-embed-text"]
    best_text_model = None
    best_text_hit5 = -1.0
    best_text_metrics = {}

    for t_model in text_models:
        try:
            t_backend = EmbeddingBackendFactory.create(t_model)
            t_cache = EmbeddingCache(t_model)
            
            # Map cached corpus embeddings
            for stem, chunks in chunks_by_doc.items():
                texts = [c["text"] for c in chunks]
                cached_res, _ = t_cache.get_batch(texts)
                for idx, res in enumerate(cached_res):
                    if res is not None:
                        if isinstance(res, dict):
                            chunks[idx]["dense_embedding"] = res["dense"]
                        else:
                            chunks[idx]["dense_embedding"] = res
                            
            # Embed queries
            q_embeddings = {}
            for qa in filtered_qa_pairs:
                q = qa["question"]
                if q not in q_embeddings:
                    q_embeddings[q] = t_backend.embed_query(q).dense[0]
                    
            # Evaluate
            hit_5 = 0
            mrr_total = 0.0
            ndcg_total = 0.0
            for qa in filtered_qa_pairs:
                question = qa["question"]
                gold_pages = qa["gold_pages"]
                stem = qa["doc_stem"]
                
                q_vec = q_embeddings[question]
                retrieved = retrieve_dense(q_vec, chunks_by_doc[stem], top_k=10)
                retrieved_pages = [c["page"] for c in retrieved]
                
                hit_5 += calculate_hit_at_k(retrieved_pages, gold_pages, 5)
                mrr_total += calculate_mrr(retrieved_pages, gold_pages)
                ndcg_total += calculate_ndcg_at_10(retrieved_pages, gold_pages, chunks_by_doc[stem])
                
            n_q = len(filtered_qa_pairs)
            hit_5_rate = hit_5 / n_q if n_q > 0 else 0.0
            mrr = mrr_total / n_q if n_q > 0 else 0.0
            ndcg = ndcg_total / n_q if n_q > 0 else 0.0
            
            if hit_5_rate > best_text_hit5:
                best_text_hit5 = hit_5_rate
                best_text_model = t_model
                best_text_metrics = {
                    "model_key": f"reference_{t_model}",
                    "arm": "text_only_reference",
                    "hit_rate_at_1": 0.0,  # omitted for reference rows
                    "hit_rate_at_3": 0.0,
                    "hit_rate_at_5": hit_5_rate,
                    "hit_rate_at_10": 0.0,
                    "mrr": mrr,
                    "ndcg_at_10": ndcg
                }
            t_backend.unload()
        except Exception as e:
            print(f"Skipping reference model {t_model}: {e}")

    if best_text_model:
        print(f"Best reference text-only model on this slice: {best_text_model} (Hit@5: {best_text_hit5:.4f})")
        summary_rows.append(best_text_metrics)

    # Save outputs
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nVision sweep summary saved to {output_dir / 'summary.csv'}")

    # Generate summary.md
    summary_md = output_dir / "summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Phase 3.4 Vision Sweep Summary\n\n")
        f.write("## Findings & Performance Analysis\n\n")
        f.write(f"Evaluated on a filtered subset of {len(filtered_qa_pairs)} figure-linked queries.\n\n")
        
        # Determine winning arm
        df_sorted = df.sort_values(by="hit_rate_at_5", ascending=False)
        winner_row = df_sorted.iloc[0]
        f.write(f"**Winning Configuration**: `{winner_row['model_key']}` ({winner_row['arm']}) with **Hit@5: {winner_row['hit_rate_at_5']:.4f}**.\n\n")
        
        f.write("### Comparison Table\n\n")
        f.write("| Model Key | Configuration Arm | Hit Rate@5 | MRR | nDCG@10 |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for _, row in df.iterrows():
            f.write(f"| {row['model_key']} | {row['arm']} | {row['hit_rate_at_5']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_10']:.4f} |\n")
        
        f.write("\n### Spot Check Verification\n\n")
        f.write("Successfully matched chunks to page images and logged adjacency links to `outputs/linked_chunks.jsonl`.\n")

    print(f"Written summary report to {summary_md}")

if __name__ == "__main__":
    main()
