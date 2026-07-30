import os
import shutil
import tempfile
import pytest
from pathlib import Path
from embedding_bench.sparse.bm25_index import BM25Index
from embedding_bench.sparse.splade_index import SpladeIndex

# Force test mode
os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"

def test_bm25_tokenization_and_search():
    # 3 documents so N=3, ensuring freq=1 terms have positive IDF
    chunks = [
        {"chunk_id": "c1", "text": "Item number 181800 is a surgical blade."},
        {"chunk_id": "c2", "text": "Distilled water with neutral pH 7 is recommended."},
        {"chunk_id": "c3", "text": "Different random document text to raise corpus size."}
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "bm25.pkl"
        index = BM25Index.load_or_build(chunks, cache_file)
        
        # Verify code preservation
        tokens = index._tokenize("Item 181800")
        assert "181800" in tokens
        
        # Search
        results = index.search("181800")
        assert results[0][0] == "c1"
        assert results[0][1] > 0.0

def test_splade_mock_indexing_and_search():
    chunks = [
        {"chunk_id": "c1", "text": "Some text for chunk 1"},
        {"chunk_id": "c2", "text": "Alternative text for chunk 2"}
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "splade.pkl"
        index = SpladeIndex.load_or_build(chunks, cache_file)
        
        results = index.search("sample query")
        assert len(results) > 0
        assert results[0][0] in ["c1", "c2"]
