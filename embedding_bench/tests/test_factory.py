import os
# Force mock mode for unit testing
os.environ["EMBEDDING_BENCH_TEST_MODE"] = "1"

import pytest
from embedding_bench.backends.factory import EmbeddingBackendFactory
from embedding_bench.backends.local_st import LocalSTBackend
from embedding_bench.backends.multi_vector import MultiVectorBackend
from embedding_bench.backends.vision_backend import VisionEmbeddingBackend

def test_factory_creation():
    # Test creating a local ST backend with correct key
    qwen = EmbeddingBackendFactory.create("qwen3-embedding-8b-fp16")
    assert isinstance(qwen, LocalSTBackend)
    assert qwen.dim == 4096
    assert qwen.max_batch == 4
    
    # Test creating a multi-vector backend
    bge = EmbeddingBackendFactory.create("bge-m3")
    assert isinstance(bge, MultiVectorBackend)
    assert bge.dim == 1024
    
    # Test creating vision backend
    granite = EmbeddingBackendFactory.create("granite-vision-embedding")
    assert isinstance(granite, VisionEmbeddingBackend)
    assert granite.dim == 128

def test_factory_unknown_key():
    with pytest.raises(KeyError, match="Unknown model_key"):
        EmbeddingBackendFactory.create("invalid-model-key")
        
    # Verify gemini-embedding is unknown/removed
    with pytest.raises(KeyError, match="Unknown model_key"):
        EmbeddingBackendFactory.create("gemini-embedding")

def test_factory_overrides():
    bge_overridden = EmbeddingBackendFactory.create("bge-m3", overrides={"device": "cpu", "max_batch": 100})
    assert bge_overridden.device == "cpu"
    assert bge_overridden.max_batch == 100
