"""Retrieval evaluation: Recall@k and MRR per chunking strategy, using a fixed
embedder held constant across strategies so only chunking quality varies."""
import json
import logging
from pathlib import Path

import numpy as np

from chunking.embedding_utils import get_default_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TOP_K = 5


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


def evaluate_strategy(strategy: str, qa_set: list[dict], embedder) -> dict:
    """Groups QA pairs by doc_stem, builds one index per doc, evaluates all
    questions for that doc against it, aggregates Recall@k and MRR across the
    whole eval set for this strategy."""
    by_doc: dict[str, list[dict]] = {}
    for qa in qa_set:
        by_doc.setdefault(qa["doc_stem"], []).append(qa)

    hits_at_k = 0
    reciprocal_ranks = []
    total = 0
    per_question_type_hits = {}
    per_question_type_total = {}

    for doc_stem, questions in by_doc.items():
        chunks = load_chunks(strategy, doc_stem)
        chunk_embeddings = build_index(chunks, embedder)
        question_embeddings = embedder.embed([q["question"] for q in questions])

        for q, q_emb in zip(questions, question_embeddings):
            ranked_idxs = retrieve_top_k(q_emb, chunk_embeddings, k=len(chunks))
            gt_indices = set(q["source_element_indices"])

            rank_of_first_hit = None
            for rank, chunk_idx in enumerate(ranked_idxs, start=1):
                chunk_source_indices = set(chunks[chunk_idx]["source_element_indices"])
                if chunk_source_indices & gt_indices:
                    rank_of_first_hit = rank
                    break

            total += 1
            qtype = q["question_type"]
            per_question_type_total[qtype] = per_question_type_total.get(qtype, 0) + 1

            if rank_of_first_hit is not None:
                reciprocal_ranks.append(1.0 / rank_of_first_hit)
                if rank_of_first_hit <= TOP_K:
                    hits_at_k += 1
                    per_question_type_hits[qtype] = per_question_type_hits.get(qtype, 0) + 1
            else:
                reciprocal_ranks.append(0.0)

    return {
        "strategy": strategy,
        "recall_at_k": hits_at_k / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "total_questions": total,
        "recall_by_question_type": {
            qt: per_question_type_hits.get(qt, 0) / per_question_type_total[qt]
            for qt in per_question_type_total
        },
    }


if __name__ == "__main__":
    with open("benchmarking/chunk_eval/qa_eval_set.json", encoding="utf-8") as f:
        qa_set = json.load(f)
    embedder = get_default_embedder()
    strategies = ["naive_baseline", "element_atomic", "section_hierarchical",
                  "geometric_grounding", "hybrid_semantic"]
    results = [evaluate_strategy(s, qa_set, embedder) for s in strategies]
    Path("benchmarking/results/stage2_chunking").mkdir(parents=True, exist_ok=True)
    with open("benchmarking/results/stage2_chunking/retrieval_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    for r in results:
        logger.info(r)
