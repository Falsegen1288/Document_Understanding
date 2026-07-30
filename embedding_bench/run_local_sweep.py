import os
# Redirection of caches to Drive D
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
from embedding_bench.cache.cost_ledger import CostLedger

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
                
            qa_pairs.append({
                "question_id": q_id,
                "doc_stem": doc_stem,
                "question": question,
                "ground_truth": ground_truth,
                "evidence": evidence,
                "gold_pages": extract_gold_pages(evidence)
            })
            
    return qa_pairs

def load_corpus_chunks(corpus_path_opt: str, strategy: str = "hybrid_semantic") -> dict[str, list[dict]]:
    doc_stems = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]
    chunks_by_doc = {}
    
    corpus_path = Path(corpus_path_opt)
    if corpus_path.is_file():
        # Load single merged JSON or JSONL file
        with open(corpus_path, "r", encoding="utf-8") as f:
            if corpus_path.suffix == ".jsonl":
                all_chunks = [json.loads(line) for line in f]
            else:
                all_chunks = json.load(f)
        
        # Group by doc stem
        for stem in doc_stems:
            chunks_by_doc[stem] = []
            
        for chunk in all_chunks:
            # Deduce stem from filename or ID
            filename = chunk.get("doc_filename", "").lower()
            chunk_id = chunk.get("chunk_id", "").lower()
            matched = False
            for stem in doc_stems:
                if stem.lower() in filename or stem.lower() in chunk_id:
                    chunks_by_doc[stem].append(chunk)
                    matched = True
                    break
            if not matched:
                # Fallback group to first stem
                chunks_by_doc[doc_stems[0]].append(chunk)
    else:
        # Load individual json files from directory
        for stem in doc_stems:
            json_file = corpus_path / f"{stem}_{strategy}.json"
            if not json_file.exists():
                # Try fallback sub-directory chunks/
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
        if chunk_vector is None:
            sim = 0.0
        else:
            sim = cosine_similarity(query_vector, chunk_vector)
        scores.append((sim, chunk))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

def calculate_mrr(retrieved_pages: list[int], gold_pages: list[int]) -> float:
    for idx, p in enumerate(retrieved_pages[:10]):
        if p in gold_pages:
            return 1.0 / (idx + 1)
    return 0.0

def calculate_hit_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    for p in retrieved_pages[:k]:
        if p in gold_pages:
            return 1.0
    return 0.0

def get_peak_memory():
    process = psutil.Process()
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return ram_mb, vram_mb

def main():
    parser = argparse.ArgumentParser(description="Phase 3.1 Local Model Sweep Harness")
    parser.add_argument("--models", required=True, help="Comma-separated model keys to evaluate")
    parser.add_argument("--corpus-path", required=True, help="Path to chunks folder or merged jsonl")
    parser.add_argument("--query-bank-path", required=True, help="Path to GT_QA_Bank.md")
    parser.add_argument("--run-id", required=True, help="Benchmark run identifier")
    parser.add_argument("--output-dir", required=True, help="Directory to save sweep results")
    parser.add_argument("--mock", action="store_true", help="Bypass model downloads and run mock embeddings")
    
    args = parser.parse_args()
    
    if args.mock:
        os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"
        print("MOCK MODE ENABLED: Bypassing model loading, using mock representations.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Ground Truth QA bank...")
    qa_pairs = parse_gt_qa_bank(Path(args.query_bank_path))
    print(f"Parsed {len(qa_pairs)} QA pairs from {args.query_bank_path}")

    print("Loading corpus chunks...")
    chunks_by_doc = load_corpus_chunks(args.corpus_path)
    total_chunks = sum(len(c) for c in chunks_by_doc.values())
    print(f"Loaded {total_chunks} chunks across documents.")

    summary_rows = []
    models_to_run = [m.strip() for m in args.models.split(",")]

    for model_key in models_to_run:
        print(f"\n=========================================")
        print(f"Starting sweep for model: {model_key}")
        print(f"=========================================")
        
        # Reset peak memory stats before loading the model
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # 1. Warm up & instantiate
        print("Instantiating model backend...")
        try:
            backend = EmbeddingBackendFactory.create(model_key)
        except Exception as e:
            print(f"Error instantiating model {model_key}: {e}. Skipping.")
            continue

        # Warm up
        backend.embed_query("warm up query")
        
        cache = EmbeddingCache(model_key)
        
        # 2. Embed Corpus
        print("Embedding corpus chunks...")
        start_corpus = time.perf_counter()
        
        cache_hits = 0
        cache_misses = 0
        total_tokens_encoded = 0
        latency_encoding_ms = 0.0
        
        for stem, chunks in chunks_by_doc.items():
            texts = [c["text"] for c in chunks]
            
            # Retrieve from cache first
            cached_results, missing_indices = cache.get_batch(texts)
            cache_hits += (len(texts) - len(missing_indices))
            cache_misses += len(missing_indices)
            
            # Map hits
            for idx, res in enumerate(cached_results):
                if res is not None:
                    if isinstance(res, dict):
                        chunks[idx]["dense_embedding"] = res["dense"]
                        chunks[idx]["sparse_embedding"] = res.get("sparse")
                        chunks[idx]["multi_vector"] = res.get("multi_vector")
                    else:
                        # Fallback for raw numpy array cached
                        chunks[idx]["dense_embedding"] = res
                        chunks[idx]["sparse_embedding"] = None
                        chunks[idx]["multi_vector"] = None
            
            # Compute misses in batch
            if missing_indices:
                missing_texts = [texts[idx] for idx in missing_indices]
                
                # Batch execution
                batch_res = backend.embed_documents(missing_texts)
                total_tokens_encoded += batch_res.token_count
                latency_encoding_ms += batch_res.latency_ms
                
                # Map and Cache misses
                for batch_idx, orig_idx in enumerate(missing_indices):
                    dense_vec = batch_res.dense[batch_idx]
                    
                    sparse_vec = batch_res.sparse[batch_idx] if batch_res.sparse else None
                    mv_vec = batch_res.multi_vector[batch_idx] if batch_res.multi_vector else None
                    
                    # Set on chunk
                    chunks[orig_idx]["dense_embedding"] = dense_vec
                    chunks[orig_idx]["sparse_embedding"] = sparse_vec
                    chunks[orig_idx]["multi_vector"] = mv_vec
                    
                    # Store to cache
                    cache.put(texts[orig_idx], {
                        "dense": dense_vec,
                        "sparse": sparse_vec,
                        "multi_vector": mv_vec
                    })
                    
        corpus_duration = (time.perf_counter() - start_corpus) * 1000.0
        throughput = len(texts) / (corpus_duration / 1000.0) if corpus_duration > 0 else 0.0
        
        # 3. Embed queries and run retrieval evaluation
        print("Embedding queries and running evaluations...")
        query_latencies = []
        
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        hit_10 = 0
        mrr_total = 0.0
        ndcg_total = 0.0
        
        # Build document indices for search
        start_index_build = time.perf_counter()
        # No heavy indexing required for flat numpy search
        index_build_duration = (time.perf_counter() - start_index_build) * 1000.0
        
        for qa in qa_pairs:
            question = qa["question"]
            gold_pages = qa["gold_pages"]
            stem = qa["doc_stem"]
            
            # Embed query
            start_q = time.perf_counter()
            q_res = backend.embed_query(question)
            q_lat = (time.perf_counter() - start_q) * 1000.0
            query_latencies.append(q_lat)
            
            q_vector = q_res.dense[0]
            
            # Search
            retrieved = retrieve_dense(q_vector, chunks_by_doc[stem], top_k=10)
            retrieved_pages = [c["page"] for c in retrieved]
            
            # Calculate metrics
            hit_1 += calculate_hit_at_k(retrieved_pages, gold_pages, 1)
            hit_3 += calculate_hit_at_k(retrieved_pages, gold_pages, 3)
            hit_5 += calculate_hit_at_k(retrieved_pages, gold_pages, 5)
            hit_10 += calculate_hit_at_k(retrieved_pages, gold_pages, 10)
            
            mrr_total += calculate_mrr(retrieved_pages, gold_pages)
            
            # nDCG calculation
            dcg = 0.0
            for idx, p in enumerate(retrieved_pages[:10]):
                rel = 1 if p in gold_pages else 0
                dcg += rel / np.log2(idx + 2)
                
            num_relevant_chunks = sum(1 for c in chunks_by_doc[stem] if c["page"] in gold_pages)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_relevant_chunks, 10)))
            
            ndcg_q = (dcg / idcg) if idcg > 0 else 0.0
            ndcg_total += ndcg_q

        # Peak RAM and VRAM footprint
        peak_ram_mb, peak_vram_mb = get_peak_memory()
        
        # Unload resources
        print("Unloading model resources...")
        backend.unload()
        
        # Calculate summary metrics
        n_queries = len(qa_pairs)
        
        hit_1_rate = hit_1 / n_queries
        hit_3_rate = hit_3 / n_queries
        hit_5_rate = hit_5 / n_queries
        hit_10_rate = hit_10 / n_queries
        mrr = mrr_total / n_queries
        ndcg = ndcg_total / n_queries
        
        latency_p50 = np.percentile(query_latencies, 50) if query_latencies else 0.0
        latency_p95 = np.percentile(query_latencies, 95) if query_latencies else 0.0
        latency_p99 = np.percentile(query_latencies, 99) if query_latencies else 0.0
        
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        
        print(f"Results for {model_key}:")
        print(f"  Hit@5: {hit_5_rate:.4f} | MRR: {mrr:.4f} | nDCG@10: {ndcg:.4f}")
        print(f"  Peak VRAM: {peak_vram_mb:.2f} MB | Peak RAM: {peak_ram_mb:.2f} MB")
        print(f"  Cache Hit Rate: {cache_hit_rate:.2%}")
        
        # Output JSON details
        result_details = {
            "model_key": model_key,
            "run_id": args.run_id,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "system_metrics": {
                "embed_corpus_latency_ms_total": corpus_duration,
                "embed_corpus_throughput_chunks_per_sec": throughput,
                "embed_query_latency_ms_p50": latency_p50,
                "embed_query_latency_ms_p95": latency_p95,
                "embed_query_latency_ms_p99": latency_p99,
                "peak_vram_mb": peak_vram_mb,
                "peak_ram_mb": peak_ram_mb,
                "index_build_time_ms": index_build_duration
            },
            "quality_metrics": {
                "hit_rate_at_1": hit_1_rate,
                "hit_rate_at_3": hit_3_rate,
                "hit_rate_at_5": hit_5_rate,
                "hit_rate_at_10": hit_10_rate,
                "mrr": mrr,
                "ndcg_at_10": ndcg
            },
            "cache_stats": {
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": cache_hit_rate
            }
        }
        
        # Save detailed JSON
        with open(output_dir / f"{model_key}.json", "w", encoding="utf-8") as f:
            json.dump(result_details, f, indent=2)
            
        # Collect summary row
        summary_rows.append({
            "model_key": model_key,
            "hit_rate_at_1": hit_1_rate,
            "hit_rate_at_3": hit_3_rate,
            "hit_rate_at_5": hit_5_rate,
            "hit_rate_at_10": hit_10_rate,
            "mrr": mrr,
            "ndcg_at_10": ndcg,
            "p50_query_latency_ms": latency_p50,
            "p95_query_latency_ms": latency_p95,
            "peak_vram_mb": peak_vram_mb,
            "peak_ram_mb": peak_ram_mb,
            "throughput_chunks_sec": throughput,
            "cache_hit_rate": cache_hit_rate
        })
        
    # Write summary CSV
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nComparative summary saved to {output_dir / 'summary.csv'}")

if __name__ == "__main__":
    main()
