import pytest
from embedding_bench.sparse.fusion import reciprocal_rank_fusion

def test_rrf_basic_equivalence():
    list_a = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    list_b = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]
    
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    
    fused_dict = dict(fused)
    assert pytest.approx(fused_dict["doc1"]) == (1.0/61 + 1.0/62)
    assert pytest.approx(fused_dict["doc2"]) == (1.0/61 + 1.0/62)
    assert pytest.approx(fused_dict["doc3"]) == (1.0/63)
    assert pytest.approx(fused_dict["doc4"]) == (1.0/63)
    
    assert fused[0][0] in ["doc1", "doc2"]
    assert fused[1][0] in ["doc1", "doc2"]
