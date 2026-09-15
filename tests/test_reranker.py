import pytest
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


def test_cross_encoder_reranker_basic():
    reranker = CrossEncoderReranker()
    query = "what is the revenue"
    passages = [
        "The sky is blue and the weather is sunny today.",
        "Revenue was $5M in Q3 according to the quarterly financial filing.",
        "The quick brown fox jumps over the lazy dog."
    ]

    ranked = reranker.rerank(query, passages, top_k=2)
    assert len(ranked) == 2
    top_idx, top_score = ranked[0]
    # Revenue passage (index 1) should be ranked first
    assert top_idx == 1
    assert top_score > ranked[1][1]


def test_cross_encoder_reranker_empty():
    reranker = CrossEncoderReranker()
    ranked = reranker.rerank("query", [])
    assert ranked == []
