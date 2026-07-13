import os
import sys
import re
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout to use UTF-8 to prevent console encode errors
sys.stdout.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv("c:/Users/user/Downloads/Document_Understanding/.env")

import yaml
import ragas
import deepeval

# Load configuration from experiment_config.yaml
config_path = Path("experiment_config.yaml")
if not config_path.exists():
    config_path = Path("c:/Users/user/Downloads/Document_Understanding/experiment_config.yaml")

with open(config_path, "r", encoding="utf-8") as f:
    experiment_config = yaml.safe_load(f)

# Environment and version validation
pinned_ragas = experiment_config["environment_pin"]["ragas"]
pinned_deepeval = experiment_config["environment_pin"]["deepeval"]
installed_ragas = ragas.__version__
installed_deepeval = deepeval.__version__

if installed_ragas != pinned_ragas:
    logger.error(f"Environment Validation Failure: Ragas version {installed_ragas} does not match pinned version {pinned_ragas}")
    sys.exit(1)
if installed_deepeval != pinned_deepeval:
    logger.error(f"Environment Validation Failure: DeepEval version {installed_deepeval} does not match pinned version {pinned_deepeval}")
    sys.exit(1)

GENERATOR_MODEL = experiment_config["generator_llm"]["model_id"]
JUDGE_MODEL = experiment_config["judge_llm"]["model_id"]
EMBEDDING_MODEL_NAME = experiment_config["embedding_model"]["name"]
TOP_K = experiment_config["retrieval"].get("top_k", 10)

logger.info(f"Environment validated successfully. Ragas={installed_ragas}, DeepEval={installed_deepeval}")
logger.info(f"Loaded config: Generator={GENERATOR_MODEL}, Judge={JUDGE_MODEL}, Embeddings={EMBEDDING_MODEL_NAME}, Top-K={TOP_K}")

# ---------------------------------------------------------------------------
# Rate-limited Groq Clients (for Generator only)
# ---------------------------------------------------------------------------
from groq import Groq, AsyncGroq

sync_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
async_groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

import sqlite3
import hashlib
from typing import Optional

GEN_CACHE_DB = "generator_cache.db"

def init_gen_cache():
    conn = sqlite3.connect(GEN_CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gen_cache (
            key TEXT PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

init_gen_cache()

def get_gen_cache(prompt: str) -> Optional[str]:
    try:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        conn = sqlite3.connect(GEN_CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM gen_cache WHERE key = ?", (prompt_hash,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception as e:
        logger.warning(f"Failed to query generator cache: {e}")
    return None

def set_gen_cache(prompt: str, response: str):
    try:
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        conn = sqlite3.connect(GEN_CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO gen_cache (key, prompt, response) VALUES (?, ?, ?)", (prompt_hash, prompt, response))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to write to generator cache: {e}")

def call_groq_with_retry(model, messages, temperature=0.0, max_retries=15, **kwargs):
    prompt_str = json.dumps(messages, sort_keys=True)
    cached = get_gen_cache(prompt_str)
    if cached is not None:
        logger.info(f"Loaded generator completion from local cache (bypassed Groq).")
        return cached

    for attempt in range(max_retries):
        try:
            # Respect rate limit of 30 RPM (roughly 1 request every 2 seconds)
            time.sleep(2.0)
            response = sync_groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            result = response.choices[0].message.content
            set_gen_cache(prompt_str, result)
            return result
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "tp_limit" in err_str:
                sleep_time = 45.0
                match = re.search(r"please try again in ([\d\.]+)s", err_str)
                if match:
                    sleep_time = float(match.group(1)) + 1.5
                logger.warning(f"Rate limited on {model}. Sleeping for {sleep_time:.2f}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Groq API error: {e}")
                raise e
    raise RuntimeError("Failed to call Groq after maximum retries due to rate limiting.")

async def call_groq_with_retry_async(model, messages, temperature=0.0, max_retries=15, **kwargs):
    prompt_str = json.dumps(messages, sort_keys=True)
    cached = get_gen_cache(prompt_str)
    if cached is not None:
        return cached

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                await asyncio.sleep(2.0)
            response = await async_groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            result = response.choices[0].message.content
            set_gen_cache(prompt_str, result)
            return result
        except Exception as e:
            err_str = str(e).lower()
            if "rate limit" in err_str or "429" in err_str or "tp_limit" in err_str:
                sleep_time = 45.0
                match = re.search(r"please try again in ([\d\.]+)s", err_str)
                if match:
                    sleep_time = float(match.group(1)) + 1.5
                logger.warning(f"Async Rate limited on {model}. Sleeping for {sleep_time:.2f}s before retry (attempt {attempt+1}/{max_retries})...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error(f"Async Groq API error: {e}")
                raise e
    raise RuntimeError("Failed to call Groq after maximum retries due to rate limiting.")

# ---------------------------------------------------------------------------
# Local Judge Pipeline Imports
# ---------------------------------------------------------------------------
from local_judge_pipeline import LocalJudgeChatModel, LocalJudgeDeepEvalLLM, score_question_consolidated

# ---------------------------------------------------------------------------
# Golden QA Bank Parser
# ---------------------------------------------------------------------------
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
            loc_type = parts[5]
            reasoning_type = parts[6]
            ans_fmt = parts[7]
            difficulty = parts[8]
            adversarial = parts[9]
            
            if not question or question == "Question":
                continue
                
            qa_pairs.append({
                "question_id": q_id,
                "doc_stem": doc_stem,
                "question": question,
                "ground_truth": ground_truth,
                "evidence": evidence,
                "location_type": loc_type,
                "reasoning_type": reasoning_type,
                "answer_format": ans_fmt,
                "difficulty": difficulty,
                "adversarial": adversarial
            })
            
    return qa_pairs

# ---------------------------------------------------------------------------
# Embedding and Retrieval Engine (BGE-M3 + RRF)
# ---------------------------------------------------------------------------
import pickle

def embed_and_build_indices(chunks: list[dict], bge_model) -> list[dict]:
    """Embed chunks in batch and enrich them with dense/sparse vectors."""
    if not chunks:
        return []
        
    # Create a unique cache key based on chunk IDs and text content
    import hashlib
    hasher = hashlib.sha256()
    for c in chunks:
        hasher.update(c.get("chunk_id", "").encode("utf-8"))
        hasher.update(c.get("text", "").encode("utf-8"))
    cache_key = hasher.hexdigest()
    
    cache_dir = Path("outputs/.embeddings_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                logger.info(f"Loaded {len(chunks)} pre-computed BGE-M3 embeddings from cache.")
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")

    texts = [c["text"] for c in chunks]
    logger.info(f"Encoding {len(chunks)} chunks with length-sorted BGE-M3 (max_length=512)...")
    
    # Sort texts by length to minimize padding overhead in batches
    indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))
    sorted_texts = [x[1] for x in indexed_texts]
    
    # Run encoding in batch with max_length=512 to prevent quadratic CPU attention scaling
    encoded = bge_model.encode(
        sorted_texts, 
        batch_size=16, 
        max_length=512, 
        return_dense=True, 
        return_sparse=True
    )
    
    # Restore original order
    dense_vecs = [None] * len(texts)
    sparse_weights = [None] * len(texts)
    
    for sorted_idx, (orig_idx, _) in enumerate(indexed_texts):
        dense_vecs[orig_idx] = encoded["dense_vecs"][sorted_idx]
        sparse_weights[orig_idx] = encoded["lexical_weights"][sorted_idx]
    
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        chunk["dense_vec"] = dense_vecs[i]
        chunk["sparse_vec"] = sparse_weights[i]
        enriched_chunks.append(chunk)
        
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(enriched_chunks, f)
    except Exception as e:
        logger.warning(f"Failed to save embeddings to cache: {e}")
        
    return enriched_chunks

def retrieve_hybrid_rrf(query: str, chunks: list[dict], bge_model, top_k=10, rrf_k=60) -> list[dict]:
    """Perform hybrid retrieval using RRF over dense and sparse representations."""
    q_encoded = bge_model.encode([query], batch_size=1, max_length=512, return_dense=True, return_sparse=True)
    q_dense = q_encoded["dense_vecs"][0]
    q_sparse = q_encoded["lexical_weights"][0]
    
    # 1. Compute dense similarity (cosine)
    from chunking.embedding_utils import cosine_similarity
    dense_scores = []
    for chunk in chunks:
        sim = cosine_similarity(q_dense, chunk["dense_vec"])
        dense_scores.append((sim, chunk))
    # Sort descending
    dense_scores.sort(key=lambda x: x[0], reverse=True)
    dense_ranks = {id(chunk): rank for rank, (_, chunk) in enumerate(dense_scores, 1)}
    
    # 2. Compute sparse similarity (lexical dot product)
    sparse_scores = []
    for chunk in chunks:
        c_sparse = chunk["sparse_vec"]
        score = 0.0
        for token, qw in q_sparse.items():
            if token in c_sparse:
                score += qw * c_sparse[token]
        sparse_scores.append((score, chunk))
    # Sort descending
    sparse_scores.sort(key=lambda x: x[0], reverse=True)
    sparse_ranks = {id(chunk): rank for rank, (_, chunk) in enumerate(sparse_scores, 1)}
    
    # 3. Apply Reciprocal Rank Fusion (RRF)
    rrf_scores = []
    for chunk in chunks:
        r_dense = dense_ranks[id(chunk)]
        r_sparse = sparse_ranks[id(chunk)]
        rrf = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_sparse))
        rrf_scores.append((rrf, chunk))
        
    # Sort by RRF score descending and return top_k
    rrf_scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in rrf_scores[:top_k]]

# ---------------------------------------------------------------------------
# Main Evaluation Harness
# ---------------------------------------------------------------------------
async def run_evaluation(strategy: str, qa_pairs: list[dict], bge_model):
    output_path = Path(f"outputs/evaluation_results_{strategy}.json")
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            if len(existing_results) == len(qa_pairs):
                logger.info(f"Strategy {strategy} already evaluated with {len(existing_results)} questions. Loading from disk.")
                return compute_averages(existing_results)
        except Exception as e:
            logger.warning(f"Failed to load existing results for {strategy}: {e}")

    logger.info(f"==================================================")
    logger.info(f"Starting Evaluation for Strategy: {strategy}")
    logger.info(f"==================================================")
    
    # 1. Load strategy chunk outputs for all 3 documents
    chunks_dir = Path("outputs/chunks")
    doc_stems = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]
    
    chunks_by_doc = {}
    total_loaded_chunks = 0
    
    for stem in doc_stems:
        chunk_file = chunks_dir / f"{stem}_{strategy}.json"
        if not chunk_file.exists():
            logger.error(f"Chunk file {chunk_file} does not exist! Please run chunking first.")
            return None
        with open(chunk_file, "r", encoding="utf-8") as f:
            doc_chunks = json.load(f)
        logger.info(f"Loaded {len(doc_chunks)} chunks for {stem}")
        
        # Build index for this doc stem
        enriched = embed_and_build_indices(doc_chunks, bge_model)
        chunks_by_doc[stem] = enriched
        total_loaded_chunks += len(enriched)
        
    logger.info(f"Total chunks loaded and indexed: {total_loaded_chunks}")
    
    # 2. Setup evaluators (use config-defined judge)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_core.embeddings import Embeddings
    
    # Wrap local BGE-M3 for Ragas
    class BGEM3LangChainEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = "BAAI/bge-m3"
            self.bge_model = model
            
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            out = self.bge_model.encode(texts, return_dense=True, return_sparse=False)
            return out['dense_vecs'].tolist()
            
        def embed_query(self, text: str) -> list[float]:
            out = self.bge_model.encode([text], return_dense=True, return_sparse=False)
            return out['dense_vecs'][0].tolist()

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

    ragas_embeddings = LangchainEmbeddingsWrapper(BGEM3LangChainEmbeddings(bge_model))
    
    # Ragas judge
    ragas_llm = LocalJudgeChatModel(model_name=JUDGE_MODEL)
    ragas_judge = LangchainLLMWrapper(ragas_llm, bypass_n=True)
    
    # DeepEval judge
    deepeval_judge = LocalJudgeDeepEvalLLM(model_name=JUDGE_MODEL)
    
    # Ragas metric instances
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    m_faith = Faithfulness(llm=ragas_judge)
    m_relevancy = AnswerRelevancy(llm=ragas_judge, embeddings=ragas_embeddings)
    m_precision = ContextPrecision(llm=ragas_judge)
    m_recall = ContextRecall(llm=ragas_judge)
    
    # DeepEval metric instances
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
    from deepeval.test_case import LLMTestCase
    
    de_faith = FaithfulnessMetric(threshold=0.5, model=deepeval_judge)
    de_relevancy = AnswerRelevancyMetric(threshold=0.5, model=deepeval_judge)
    de_precision = ContextualPrecisionMetric(threshold=0.5, model=deepeval_judge)
    de_recall = ContextualRecallMetric(threshold=0.5, model=deepeval_judge)
    
    results = []
    judge_tasks = []
    
    # Process QA pairs sequentially for generation (with 4.5s delay to respect Groq TPM limit)
    # but run local judge evaluations concurrently in the background.
    for i, qa in enumerate(qa_pairs, 1):
        q_id = qa["question_id"]
        stem = qa["doc_stem"]
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        
        # 1. Retrieval
        relevant_chunks = chunks_by_doc[stem]
        retrieved = retrieve_hybrid_rrf(question, relevant_chunks, bge_model, top_k=10)
        
        # Dynamically limit retrieved chunks using tiktoken to stay strictly under 4,800 tokens total
        # to guarantee we remain under the 6,000 TPM Groq limit
        retrieved_texts = [c["text"] for c in retrieved]
        import tiktoken
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoding = None
            
        def get_total_tokens(texts):
            context_str_temp = "\n\n".join(texts)
            # System prompt + template + question is roughly 150 tokens
            if encoding:
                return len(encoding.encode(context_str_temp)) + len(encoding.encode(question)) + 150
            else:
                return (len(context_str_temp) + len(question)) // 3 + 150

        while len(retrieved_texts) > 1 and get_total_tokens(retrieved_texts) > 4800:
            retrieved_texts.pop()
            retrieved.pop()
            
        context_str = "\n\n".join(retrieved_texts)
        
        # 2. Generation: Use config-defined generator
        gen_messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question using only the provided context. If the context does not contain the answer, say 'I don't know'. Do not make up facts."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion:\n{question}\n\nAnswer:"}
        ]
        
        prompt_str = json.dumps(gen_messages, sort_keys=True)
        was_cached = get_gen_cache(prompt_str) is not None
        
        logger.info(f"[{i}/{len(qa_pairs)}] Generating answer for {q_id} with {GENERATOR_MODEL}...")
        generated_answer = await call_groq_with_retry_async(model=GENERATOR_MODEL, messages=gen_messages, temperature=0.0)
        logger.info(f"[{i}/{len(qa_pairs)}] Generated answer for {q_id}: {generated_answer[:100]}...")
        
        # Helper to run judge evaluation in background and append to results
        async def evaluate_judge_and_store(q_id_val, stem_val, question_val, ground_truth_val, generated_answer_val, context_str_val, retrieved_chunks_val):
            logger.info(f"Running consolidated metrics scoring for {q_id_val}...")
            scores = await score_question_consolidated(
                question=question_val,
                context=context_str_val,
                response=generated_answer_val,
                ground_truth=ground_truth_val
            )
            
            score_faith = scores.get("ragas_faithfulness")
            score_relevancy = scores.get("ragas_answer_relevancy")
            score_precision = scores.get("ragas_context_precision")
            score_recall = scores.get("ragas_context_recall")
            
            de_score_faith = scores.get("deepeval_faithfulness")
            de_score_relevancy = scores.get("deepeval_answer_relevancy")
            de_score_precision = scores.get("deepeval_contextual_precision")
            de_score_recall = scores.get("deepeval_contextual_recall")
            
            result_item = {
                "question_id": q_id_val,
                "doc_stem": stem_val,
                "question": question_val,
                "ground_truth": ground_truth_val,
                "generated_answer": generated_answer_val,
                "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunks_val],
                "scores": {
                    "ragas": {
                        "faithfulness": score_faith,
                        "answer_relevancy": score_relevancy,
                        "context_precision": score_precision,
                        "context_recall": score_recall
                    },
                    "deepeval": {
                        "faithfulness": de_score_faith,
                        "answer_relevancy": de_score_relevancy,
                        "contextual_precision": de_score_precision,
                        "contextual_recall": de_score_recall
                    }
                }
            }
            results.append(result_item)
            logger.info(f"Scores for {q_id_val}: Ragas [F={score_faith}, AR={score_relevancy}, CP={score_precision}, CR={score_recall}] | DeepEval [F={de_score_faith}, AR={de_score_relevancy}, CP={de_score_precision}, CR={de_score_recall}]")
            
        # Launch judge scoring task in background
        judge_tasks.append(asyncio.create_task(evaluate_judge_and_store(
            q_id, stem, question, ground_truth, generated_answer, context_str, retrieved
        )))
        
        # Respect Groq token rate limit by sleeping exactly 9.0 seconds after launching the task
        # only if we actually queried the API (i.e. not cached).
        if not was_cached:
            await asyncio.sleep(9.0)
        
    # Await all background judge tasks
    await asyncio.gather(*judge_tasks)
        
    # 4. Save results and compute overall stats
    output_path = Path(f"outputs/evaluation_results_{strategy}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed results to {output_path}")
    
    # Compute averages
    stats = compute_averages(results)
    return stats

def compute_averages(results: list[dict]) -> dict:
    metrics = {
        "ragas_faithfulness": [],
        "ragas_answer_relevancy": [],
        "ragas_context_precision": [],
        "ragas_context_recall": [],
        "deepeval_faithfulness": [],
        "deepeval_answer_relevancy": [],
        "deepeval_contextual_precision": [],
        "deepeval_contextual_recall": []
    }
    
    for r in results:
        scores = r["scores"]
        
        # Ragas
        if scores["ragas"]["faithfulness"] is not None:
            metrics["ragas_faithfulness"].append(scores["ragas"]["faithfulness"])
        if scores["ragas"]["answer_relevancy"] is not None:
            metrics["ragas_answer_relevancy"].append(scores["ragas"]["answer_relevancy"])
        if scores["ragas"]["context_precision"] is not None:
            metrics["ragas_context_precision"].append(scores["ragas"]["context_precision"])
        if scores["ragas"]["context_recall"] is not None:
            metrics["ragas_context_recall"].append(scores["ragas"]["context_recall"])
            
        # DeepEval
        if scores["deepeval"]["faithfulness"] is not None:
            metrics["deepeval_faithfulness"].append(scores["deepeval"]["faithfulness"])
        if scores["deepeval"]["answer_relevancy"] is not None:
            metrics["deepeval_answer_relevancy"].append(scores["deepeval"]["answer_relevancy"])
        if scores["deepeval"]["contextual_precision"] is not None:
            metrics["deepeval_contextual_precision"].append(scores["deepeval"]["contextual_precision"])
        if scores["deepeval"]["contextual_recall"] is not None:
            metrics["deepeval_contextual_recall"].append(scores["deepeval"]["contextual_recall"])
            
    averages = {}
    for name, vals in metrics.items():
        averages[name] = sum(vals) / len(vals) if vals else 0.0
        
    return averages

# ---------------------------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Stage 2 A4 Chunking Evaluation Harness")
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["naive_baseline", "element_atomic", "section_hierarchical", "geometric_grounding", "hybrid_semantic", "all"],
        help="Chunking strategy to evaluate or 'all'"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of QA pairs to evaluate"
    )
    args = parser.parse_args()
    
    qa_filepath = Path("C:/Users/user/Downloads/GT_QA_Bank.md")
    if not qa_filepath.exists():
        logger.error(f"QA Bank filepath {qa_filepath} does not exist!")
        return
        
    qa_pairs = parse_gt_qa_bank(qa_filepath)
    if args.limit:
        qa_pairs = qa_pairs[:args.limit]
        logger.info(f"Limiting evaluation to the first {len(qa_pairs)} QA pairs.")
    else:
        logger.info(f"Successfully parsed {len(qa_pairs)} QA pairs from golden bank.")
    
    # Load BGE-M3 locally
    from FlagEmbedding import BGEM3FlagModel
    logger.info("Loading BGE-M3 model...")
    bge_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    
    strategies = [args.strategy] if args.strategy != "all" else ["naive_baseline", "element_atomic", "section_hierarchical", "geometric_grounding", "hybrid_semantic"]
    
    all_stats = {}
    for strat in strategies:
        stats = await run_evaluation(strat, qa_pairs, bge_model)
        if stats:
            all_stats[strat] = stats
            
    # Print summary table
    logger.info("\n==================================================")
    logger.info("FINAL COMPARATIVE BENCHMARK SUMMARY")
    logger.info("==================================================")
    header = f"{'Strategy':<22} | {'R-Faith':<7} | {'R-Rel':<7} | {'R-Prec':<7} | {'R-Rec':<7} | {'D-Faith':<7} | {'D-Rel':<7} | {'D-Prec':<7} | {'D-Rec':<7}"
    logger.info(header)
    logger.info("-" * len(header))
    
    for strat, s in all_stats.items():
        logger.info(
            f"{strat:<22} | "
            f"{s['ragas_faithfulness']:.4f} | "
            f"{s['ragas_answer_relevancy']:.4f} | "
            f"{s['ragas_context_precision']:.4f} | "
            f"{s['ragas_context_recall']:.4f} | "
            f"{s['deepeval_faithfulness']:.4f} | "
            f"{s['deepeval_answer_relevancy']:.4f} | "
            f"{s['deepeval_contextual_precision']:.4f} | "
            f"{s['deepeval_contextual_recall']:.4f}"
        )
        
    # Write summary stats file
    summary_path = Path("outputs/evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved overall summary statistics to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main())
