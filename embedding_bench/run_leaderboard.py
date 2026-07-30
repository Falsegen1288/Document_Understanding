import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, "D:/Downloads/Document_Understanding")
sys.path.insert(0, "c:/Users/user/Downloads/Document_Understanding")

from embedding_bench.run_vision_sweep import parse_gt_qa_bank, load_corpus_chunks, extract_gold_pages, verify_and_correct_gold_pages
from embedding_bench.run_hybrid_sweep import is_spec_query
from embedding_bench.sparse.bm25_index import BM25Index
from embedding_bench.sparse.splade_index import SpladeIndex
from embedding_bench.backends.factory import EmbeddingBackendFactory
from embedding_bench.cache.embedding_cache import EmbeddingCache
from embedding_bench.leaderboard import eval_generation_metrics, is_judge_online, build_judge_sample

# Use standard utf-8 console output for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

async def main_async():
    parser = argparse.ArgumentParser(description="Phase 3.5 Consolidated Leaderboard Generator")
    parser.add_argument("--result-dirs", default="outputs/benchmark_runs/test_run,outputs/benchmark_runs/test_run_hybrid,outputs/benchmark_runs/vision_sweep_v1", help="Comma-separated result folders")
    parser.add_argument("--output", default="outputs/benchmark_runs/final_leaderboard.csv", help="Leaderboard output CSV path")
    parser.add_argument("--top-n", type=int, default=19, help="Print top N models")
    parser.add_argument("--mock", action="store_true", help="Use mock embeddings and skip real judge server")
    
    args = parser.parse_args()
    
    if args.mock:
        os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"
        print("MOCK MODE ENABLED")
    else:
        if not is_judge_online():
            raise RuntimeError(
                "Pre-flight Health Check Failed: Local judge server (Ollama) is offline at http://localhost:8000/v1. "
                "Refusing to execute sweep without an active judge server unless --mock flag is explicitly passed."
            )
    
    res_dirs = [Path(d.strip()) for d in args.result_dirs.split(",")]
    local_dir = res_dirs[0]
    hybrid_dir = res_dirs[1]
    vision_dir = res_dirs[2]
    
    # Check that required sweep folders exist
    if not local_dir.exists() or not hybrid_dir.exists() or not vision_dir.exists():
        # Fall back to default names if mismatched
        local_dir = Path("outputs/benchmark_runs/test_run")
        hybrid_dir = Path("outputs/benchmark_runs/test_run_hybrid")
        vision_dir = Path("outputs/benchmark_runs/vision_sweep_v1")

    print("Parsing golden QA bank...")
    qa_bank_path = Path("D:/Downloads/GT_QA_Bank.md")
    if not qa_bank_path.exists():
        qa_bank_path = Path("C:/Users/user/Downloads/GT_QA_Bank.md")
    qa_pairs = parse_gt_qa_bank(qa_bank_path)
    qa_pairs = verify_and_correct_gold_pages(qa_pairs)

    
    print("Loading corpus chunks...")
    chunks_by_doc = load_corpus_chunks("outputs/chunks")
    
    # Load/Build BM25 and SPLADE indexes
    cache_dir = Path("outputs/.sparse_index_cache")
    bm25_indices = {}
    splade_indices = {}
    for stem, chunks in chunks_by_doc.items():
        bm25_indices[stem] = BM25Index.load_or_build(chunks, cache_dir / f"bm25_index_{stem}.pkl")
        splade_indices[stem] = SpladeIndex.load_or_build(chunks, cache_dir / f"splade_index_{stem}.pkl")

    # Load spatial chunks mapping for vision models
    linked_map = {}
    linked_chunks_file = Path("outputs/linked_chunks.jsonl")
    if linked_chunks_file.exists():
        with open(linked_chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                linked_map[data["chunk_id"]] = data

    # Separate queries into overall vs figure-linked subset
    filtered_qa_pairs = []
    for qa in qa_pairs:
        stem = qa["doc_stem"]
        gold_pages = qa["gold_pages"]
        has_linked_fig = False
        for chunk in chunks_by_doc[stem]:
            if chunk["page"] in gold_pages:
                lc = linked_map.get(chunk["chunk_id"])
                if lc and lc.get("figure_image_path") is not None:
                    has_linked_fig = True
                    break
        if has_linked_fig:
            filtered_qa_pairs.append(qa)

    # Read system metadata and VRAM stats from the JSON logs
    system_metadata = {}
    json_paths = list(local_dir.glob("*.json"))
    for jp in json_paths:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
                m_key = data["model_key"]
                system_metadata[m_key] = data.get("system_metrics", {})
        except Exception:
            pass

    # Read retrieval metrics from previous sweeps
    hybrid_df = pd.read_csv(hybrid_dir / "summary.csv")
    vision_df = pd.read_csv(vision_dir / "summary.csv")
    
    leaderboard_rows = []

    print("\nStarting generation evaluations for all experimental arms...")

    # Define all candidate models we ran
    candidate_models = ["bge-m3", "qwen3-embedding-8b-4bit", "nv-embed-v2-fp16", "nomic-embed-text"]

    # 1. Baselines
    for baseline in ["bm25_only", "splade_only"]:
        print(f"Evaluating Baseline generation metrics: {baseline}...")
        gen_avg = await eval_generation_metrics(
            arm=baseline,
            model_key="baseline",
            chunks_by_doc=chunks_by_doc,
            qa_pairs=qa_pairs,
            bm25_indices=bm25_indices,
            splade_indices=splade_indices,
            query_embeddings={}
        )
        
        # Merge with retrieval metrics
        row = hybrid_df[hybrid_df["arm_key"] == baseline].iloc[0]
        leaderboard_rows.append({
            "model_key": "baseline",
            "arm": baseline,
            "sparse_method": "bm25" if baseline == "bm25_only" else "splade",
            "hit_rate_at_1": row["hit_rate_at_1"],
            "hit_rate_at_3": row["hit_rate_at_3"],
            "hit_rate_at_5": row["hit_rate_at_5"],
            "hit_rate_at_10": row["hit_rate_at_10"],
            "mrr": row["mrr"],
            "ndcg_at_10": row["ndcg_at_10"],
            "spec_hit_rate_at_5": row["spec_hit_rate_at_5"],
            "spec_mrr": row["spec_mrr"],
            "spec_ndcg_at_10": row["spec_ndcg_at_10"],
            "faithfulness": gen_avg.get("ragas_faithfulness", 0.0),
            "answer_relevancy": gen_avg.get("ragas_answer_relevancy", 0.0),
            "context_precision": gen_avg.get("ragas_context_precision", 0.0),
            "context_recall": gen_avg.get("ragas_context_recall", 0.0),
            "embed_corpus_latency_ms_total": 0.0,
            "embed_query_latency_ms_p95": 0.0,
            "peak_vram_mb": 0.0,
            "cost_usd_total": 0.0,
            "cost_per_1k_queries_estimate": 0.0
        })

    # 2. Text & Hybrid models
    for model_key in candidate_models:
        # Load embeddings in chunks for search retrieval
        cache = EmbeddingCache(model_key)
        for stem, chunks in chunks_by_doc.items():
            texts = [c["text"] for c in chunks]
            cached_res, _ = cache.get_batch(texts)
            for idx, res in enumerate(cached_res):
                if res is not None:
                    if isinstance(res, dict):
                        chunks[idx]["dense_embedding"] = res["dense"]
                        chunks[idx]["sparse_embedding"] = res.get("sparse")
                        chunks[idx]["multi_vector"] = res.get("multi_vector")
                    else:
                        chunks[idx]["dense_embedding"] = res

        # Embed queries using Factory
        backend = EmbeddingBackendFactory.create(model_key)
        query_embeddings = {}
        for qa in qa_pairs:
            q = qa["question"]
            if q not in query_embeddings:
                query_embeddings[q] = backend.embed_query(q)

        # Get VRAM and Latency system metrics
        sys_info = system_metadata.get(model_key, {})
        vram = sys_info.get("peak_vram_mb", 0.0)
        corpus_lat = sys_info.get("embed_corpus_latency_ms_total", 0.0)
        query_lat_p95 = sys_info.get("embed_query_latency_ms_p95", 0.0)

        # Define arms
        model_arms = ["dense_only", "bm25_hybrid", "splade_hybrid"]
        if model_key == "bge-m3":
            model_arms.append("dense_sparse_colbert_hybrid")

        for arm in model_arms:
            arm_key = f"{model_key}_{arm}"
            print(f"Evaluating model arm generation metrics: {arm_key}...")
            
            gen_avg = await eval_generation_metrics(
                arm=arm_key,
                model_key=model_key,
                chunks_by_doc=chunks_by_doc,
                qa_pairs=qa_pairs,
                bm25_indices=bm25_indices,
                splade_indices=splade_indices,
                query_embeddings=query_embeddings
            )
            
            row = hybrid_df[hybrid_df["arm_key"] == arm_key].iloc[0]
            leaderboard_rows.append({
                "model_key": model_key,
                "arm": arm,
                "sparse_method": "none" if arm == "dense_only" else ("bm25" if "bm25" in arm else ("splade" if "splade" in arm else "multi_vector")),
                "hit_rate_at_1": row["hit_rate_at_1"],
                "hit_rate_at_3": row["hit_rate_at_3"],
                "hit_rate_at_5": row["hit_rate_at_5"],
                "hit_rate_at_10": row["hit_rate_at_10"],
                "mrr": row["mrr"],
                "ndcg_at_10": row["ndcg_at_10"],
                "spec_hit_rate_at_5": row["spec_hit_rate_at_5"],
                "spec_mrr": row["spec_mrr"],
                "spec_ndcg_at_10": row["spec_ndcg_at_10"],
                "faithfulness": gen_avg.get("ragas_faithfulness", 0.0),
                "answer_relevancy": gen_avg.get("ragas_answer_relevancy", 0.0),
                "context_precision": gen_avg.get("ragas_context_precision", 0.0),
                "context_recall": gen_avg.get("ragas_context_recall", 0.0),
                "embed_corpus_latency_ms_total": corpus_lat,
                "embed_query_latency_ms_p95": query_lat_p95,
                "peak_vram_mb": vram,
                "cost_usd_total": 0.0,
                "cost_per_1k_queries_estimate": 0.0
            })

        backend.unload()

    # 3. Vision / Multimodal models
    if args.mock:
        os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"
    vision_models = ["granite-vision-embedding", "qwen3-vl"]
    for model_key in vision_models:
        backend = EmbeddingBackendFactory.create(model_key)
        
        # Query embeddings
        query_embeddings = {}
        for qa in filtered_qa_pairs:
            q = qa["question"]
            if q not in query_embeddings:
                query_embeddings[q] = backend.embed_query(q)

        # Retrieve system metrics
        vram = 0.0
        corpus_lat = 0.0
        query_lat_p95 = 0.0
        
        # Check if logs exist
        v_log_path = vision_dir / f"{model_key}.json"
        if v_log_path.exists():
            try:
                with open(v_log_path, "r", encoding="utf-8") as f:
                    v_data = json.load(f)
                    sys_v = v_data.get("system_metrics", {})
                    vram = sys_v.get("peak_vram_mb", 0.0)
                    corpus_lat = sys_v.get("embed_corpus_latency_ms_total", 0.0)
                    query_lat_p95 = sys_v.get("embed_query_latency_ms_p95", 0.0)
            except Exception:
                pass

        for arm in ["text_only", "text_plus_image"]:
            print(f"Evaluating vision arm generation metrics: {model_key}_{arm}...")
            
            # Map vision embeddings to chunks based on arm configuration
            for stem, chunks in chunks_by_doc.items():
                current_linked_chunks = []
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
                            "figure_image_path": lc.get("figure_image_path") if lc else None
                        })
                corpus_res = backend.embed_documents(current_linked_chunks)
                for idx, chunk in enumerate(chunks):
                    chunk["dense_embedding"] = corpus_res.dense[idx]

            # Run evaluation only on the 22 figure-linked queries
            gen_avg = await eval_generation_metrics(
                arm=arm,
                model_key=model_key,
                chunks_by_doc=chunks_by_doc,
                qa_pairs=filtered_qa_pairs,
                bm25_indices=bm25_indices,
                splade_indices=splade_indices,
                query_embeddings=query_embeddings,
                linked_map=linked_map
            )
            
            row = vision_df[(vision_df["model_key"] == model_key) & (vision_df["arm"] == arm)].iloc[0]
            # Compute real spec_lookup slice metrics for vision models
            # using the corrected is_spec_query classifier
            from embedding_bench.run_hybrid_sweep import calculate_hit_at_k as calc_hit_k
            from embedding_bench.run_hybrid_sweep import calculate_mrr as calc_mrr_fn
            from embedding_bench.run_hybrid_sweep import calculate_ndcg_at_10 as calc_ndcg
            from embedding_bench.leaderboard import cosine_similarity, retrieve_dense as lb_retrieve_dense
            
            spec_qa_in_slice = [qa for qa in filtered_qa_pairs if is_spec_query(qa["question"])]
            n_spec_in_slice = len(spec_qa_in_slice)
            
            if n_spec_in_slice > 0:
                spec_hit5_total = 0.0
                spec_mrr_total = 0.0
                spec_ndcg_total = 0.0
                for qa in spec_qa_in_slice:
                    q_vec = query_embeddings.get(qa["question"])
                    if q_vec is not None:
                        q_dense = q_vec.dense[0]
                    else:
                        q_dense = None
                    stem = qa["doc_stem"]
                    gold_pages = qa["gold_pages"]
                    if q_dense is not None:
                        retrieved_chunks = lb_retrieve_dense(q_dense, chunks_by_doc[stem], top_k=10)
                        retrieved_ids = [cid for cid, _ in retrieved_chunks[:10]]
                    else:
                        retrieved_ids = []
                    chunk_map = {c["chunk_id"]: c["page"] for c in chunks_by_doc[stem]}
                    retrieved_pages = [chunk_map[cid] for cid in retrieved_ids if cid in chunk_map]
                    spec_hit5_total += calc_hit_k(retrieved_pages, gold_pages, 5)
                    spec_mrr_total += calc_mrr_fn(retrieved_pages, gold_pages)
                    spec_ndcg_total += calc_ndcg(retrieved_pages, gold_pages, chunks_by_doc[stem])
                vision_spec_hit5 = spec_hit5_total / n_spec_in_slice
                vision_spec_mrr = spec_mrr_total / n_spec_in_slice
                vision_spec_ndcg = spec_ndcg_total / n_spec_in_slice
            else:
                vision_spec_hit5 = float('nan')
                vision_spec_mrr = float('nan')
                vision_spec_ndcg = float('nan')
            
            # Flag statistical reliability
            if n_spec_in_slice < 5:
                print(f"    WARNING: Only {n_spec_in_slice} spec_lookup queries in the figure-linked slice — spec metrics are statistically unreliable.")
            
            leaderboard_rows.append({
                "model_key": model_key,
                "arm": arm,
                "sparse_method": "none",
                "hit_rate_at_1": row["hit_rate_at_1"],
                "hit_rate_at_3": row["hit_rate_at_3"],
                "hit_rate_at_5": row["hit_rate_at_5"],
                "hit_rate_at_10": row["hit_rate_at_10"],
                "mrr": row["mrr"],
                "ndcg_at_10": row["ndcg_at_10"],
                "spec_hit_rate_at_5": vision_spec_hit5,
                "spec_mrr": vision_spec_mrr,
                "spec_ndcg_at_10": vision_spec_ndcg,
                "n_spec_queries_in_slice": n_spec_in_slice,
                "faithfulness": gen_avg.get("ragas_faithfulness", 0.0),
                "answer_relevancy": gen_avg.get("ragas_answer_relevancy", 0.0),
                "context_precision": gen_avg.get("ragas_context_precision", 0.0),
                "context_recall": gen_avg.get("ragas_context_recall", 0.0),
                "embed_corpus_latency_ms_total": corpus_lat,
                "embed_query_latency_ms_p95": query_lat_p95,
                "peak_vram_mb": vram,
                "cost_usd_total": 0.0,
                "cost_per_1k_queries_estimate": 0.0
            })

        backend.unload()

    # Compile DataFrame
    df = pd.DataFrame(leaderboard_rows)

    # Clean duplicates and format columns
    df = df.drop_duplicates(subset=["model_key", "arm"])

    # Sorting priority:
    # 1. spec_hit_rate_at_5 descending
    # 2. faithfulness descending
    # 3. embed_query_latency_ms_p95 ascending
    # 4. peak_vram_mb ascending
    df = df.sort_values(
        by=["spec_hit_rate_at_5", "faithfulness", "embed_query_latency_ms_p95", "peak_vram_mb"],
        ascending=[False, False, True, True]
    )

    # Save to output CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFinal Leaderboard saved to {output_path}")

    # Print top N configurations
    print(f"\n================ TOP {args.top_n} CONFIGURATIONS ================")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(args.top_n)[["model_key", "arm", "spec_hit_rate_at_5", "hit_rate_at_5", "faithfulness", "embed_query_latency_ms_p95", "peak_vram_mb"]])
    print("===============================================================\n")

    # Generate leaderboard_summary.md
    winner = df.iloc[0]
    runner_up = df.iloc[1]
    
    summary_md_path = output_path.parent / "leaderboard_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Consolidated Leaderboard Summary\n\n")
        f.write("## Selection Recommendation\n\n")
        f.write(f"- **Winning Configuration**: `{winner['model_key']}` with configuration `{winner['arm']}`.\n")
        f.write(f"- **Runner-Up Configuration**: `{runner_up['model_key']}` with configuration `{runner_up['arm']}`.\n\n")
        
        f.write("### Metric Deltas\n\n")
        f.write(f"- **Spec Hit Rate@5 delta**: {winner['spec_hit_rate_at_5']:.4f} (winner) vs. {runner_up['spec_hit_rate_at_5']:.4f} (runner-up) [Delta: {winner['spec_hit_rate_at_5'] - runner_up['spec_hit_rate_at_5']:.4f}]\n")
        f.write(f"- **Faithfulness delta**: {winner['faithfulness']:.4f} vs. {runner_up['faithfulness']:.4f} [Delta: {winner['faithfulness'] - runner_up['faithfulness']:.4f}]\n")
        f.write(f"- **Query Latency (p95) delta**: {winner['embed_query_latency_ms_p95']:.2f}ms vs. {runner_up['embed_query_latency_ms_p95']:.2f}ms\n\n")
        
        f.write("### Cost Confirmation\n\n")
        f.write("- **Total API Spend**: **$0.00** (All models evaluated are local open-source and free to run).\n")

    print(f"Summary report written to {summary_md_path}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
