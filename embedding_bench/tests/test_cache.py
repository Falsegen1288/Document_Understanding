import os
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
from embedding_bench.cache.embedding_cache import EmbeddingCache
from embedding_bench.cache.cost_ledger import CostLedger

def test_cache_roundtrip_and_partial_hits():
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = EmbeddingCache("test-model", cache_root=temp_dir)
        
        # Test put/get for dense representation
        text_1 = "hello world"
        emb_1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cache.put(text_1, emb_1)
        
        retrieved_1 = cache.get(text_1)
        assert retrieved_1 is not None
        assert np.allclose(retrieved_1, emb_1)
        
        # Test put/get for sparse/multi_vector structures
        text_2 = "complex text"
        payload_2 = {
            "dense": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "sparse": {100: 0.9},
            "multi_vector": [np.array([[0.1, 0.2]], dtype=np.float32)]
        }
        cache.put(text_2, payload_2)
        
        retrieved_2 = cache.get(text_2)
        assert retrieved_2 is not None
        assert np.allclose(retrieved_2["dense"], payload_2["dense"])
        assert retrieved_2["sparse"] == payload_2["sparse"]
        assert np.allclose(retrieved_2["multi_vector"][0], payload_2["multi_vector"][0])
        
        # Test get_batch and missing indices
        texts = ["hello world", "not cached text", "complex text"]
        results, missing = cache.get_batch(texts)
        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None
        assert missing == [1]
        
        # Test clear
        EmbeddingCache.clear("test-model", cache_root=temp_dir)
        assert cache.get(text_1) is None

def test_cost_ledger():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_file = Path(temp_dir) / "cost_ledger.db"
        ledger = CostLedger(str(db_file))
        
        # Log 3 calls
        ledger.log_call("model-a", 1000, 0.05, "run-1", "embed_documents")
        ledger.log_call("model-a", 500, 0.025, "run-1", "embed_query")
        ledger.log_call("model-b", 2000, 0.10, "run-1", "embed_documents")
        
        # Check totals
        assert ledger.get_total() == pytest.approx(0.175)
        assert ledger.get_total("model-a") == pytest.approx(0.075)
        assert ledger.get_total("model-b") == pytest.approx(0.10)
        assert ledger.get_total("unknown-model") == pytest.approx(0.0)
        
        # Check summary
        summary = ledger.get_run_summary("run-1")
        assert summary["model-a"] == pytest.approx(0.075)
        assert summary["model-b"] == pytest.approx(0.10)
