"""LLM-graded answer correctness, run only on successfully-retrieved questions.
Reuses the Step 0 LLM client for both generation and judging in a single combined call
to minimize API requests and avoid rate limits."""
import os
import sys
import time
# Add project root to path for imports
sys.path.append(os.path.abspath('c:/Users/user/Downloads/Document_Understanding'))

import json
import logging
from pathlib import Path
from groq import Groq
import numpy as np

from chunking.embedding_utils import get_default_embedder
from algorithms.config import GROQ_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TOP_K = 5

COMBINED_PROMPT = """You are a QA grading assistant.
First, answer the question below using ONLY the provided context. Make the answer concise.
Second, compare your generated answer to the reference answer.
Third, judge if your candidate answer conveys the same factual information as the reference answer.

Context:
{context}

Question: {question}
Reference Answer: {reference}

Respond in strict JSON format, with no markdown fences, no preamble, and no postamble:
{{
  "candidate_answer": "...",
  "correct": true or false,
  "reasoning": "one sentence explaining why it is correct or incorrect"
}}"""


def load_chunks(strategy: str, doc_stem: str) -> list[dict]:
    path = Path(f"outputs/chunks/{doc_stem}_{strategy}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict], embedder) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    return embedder.embed(texts)


def retrieve_top_k(query_embedding: np.ndarray, chunk_embeddings: np.ndarray, k: int) -> list[int]:
    norms = np.linalg.norm(chunk_embeddings, axis=1) * (np.linalg.norm(query_embedding) + 1e-8)
    norms[norms == 0] = 1e-8
    sims = (chunk_embeddings @ query_embedding) / norms
    return list(np.argsort(-sims)[:k])


def generate_and_grade(strategy: str, qa_set: list[dict], embedder, llm_call_fn) -> dict:
    by_doc: dict[str, list[dict]] = {}
    for qa in qa_set:
        by_doc.setdefault(qa["doc_stem"], []).append(qa)

    total_graded = 0
    correct_graded = 0
    graded_samples = []

    for doc_stem, questions in by_doc.items():
        chunks = load_chunks(strategy, doc_stem)
        chunk_embeddings = build_index(chunks, embedder)
        question_embeddings = embedder.embed([q["question"] for q in questions])

        for q, q_emb in zip(questions, question_embeddings):
            ranked_idxs = retrieve_top_k(q_emb, chunk_embeddings, k=TOP_K)
            gt_indices = set(q["source_element_indices"])

            # Check if it is a hit in TOP_K
            is_hit = False
            retrieved_chunk_texts = []
            for rank, chunk_idx in enumerate(ranked_idxs, start=1):
                chunk_source_indices = set(chunks[chunk_idx]["source_element_indices"])
                if chunk_source_indices & gt_indices:
                    is_hit = True
                retrieved_chunk_texts.append(chunks[chunk_idx]["text"])

            if not is_hit:
                # Skip if retrieval failed
                continue

            # Build context from retrieved chunks
            context = "\n---\n".join(retrieved_chunk_texts)
            
            # Call LLM
            prompt = COMBINED_PROMPT.format(context=context, question=q["question"], reference=q["reference_answer"])
            try:
                # Add delay to avoid rate limit (30 RPM -> 1 request per 2.2 seconds is safe)
                time.sleep(2.2)
                
                raw = llm_call_fn(prompt)
                
                # Clean JSON markdown blocks if any
                cleaned = raw.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned.removeprefix("```json")
                if cleaned.endswith("```"):
                    cleaned = cleaned.removesuffix("```")
                cleaned = cleaned.strip()
                
                parsed = json.loads(cleaned)
                candidate_answer = parsed.get("candidate_answer", "")
                is_correct = parsed.get("correct", False)
                reasoning = parsed.get("reasoning", "")
                
                total_graded += 1
                if is_correct:
                    correct_graded += 1

                graded_samples.append({
                    "question": q["question"],
                    "reference_answer": q["reference_answer"],
                    "candidate_answer": candidate_answer,
                    "is_correct": is_correct,
                    "reasoning": reasoning,
                    "retrieved_context_snippet": context[:300] + "..."
                })
                logger.info(f"[{strategy}] Graded Q: '{q['question']}' -> Correct: {is_correct}")
            except Exception as e:
                logger.warning(f"Grading failed for question: '{q['question']}': {e}")

    accuracy = correct_graded / total_graded if total_graded else 0.0
    return {
        "strategy": strategy,
        "total_hits_graded": total_graded,
        "correct_answers": correct_graded,
        "correctness_accuracy": accuracy,
        "samples": graded_samples
    }


def main():
    client = Groq(api_key=GROQ_API_KEY)
    
    def llm_call(prompt: str) -> str:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    with open("benchmarking/chunk_eval/qa_eval_set.json", encoding="utf-8") as f:
        qa_set = json.load(f)
    embedder = get_default_embedder()
    strategies = ["naive_baseline", "element_atomic", "section_hierarchical",
                  "geometric_grounding", "hybrid_semantic"]
                  
    results = []
    for s in strategies:
        logger.info(f"Starting Answer Grading for {s}...")
        res = generate_and_grade(s, qa_set, embedder, llm_call)
        results.append(res)

    out_path = Path("benchmarking/results/stage2_chunking/answer_grading.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    for r in results:
        print(f"Strategy: {r['strategy']} | Graded: {r['total_hits_graded']} | Correct: {r['correct_answers']} | Accuracy: {r['correctness_accuracy']:.2%}")


if __name__ == "__main__":
    main()
