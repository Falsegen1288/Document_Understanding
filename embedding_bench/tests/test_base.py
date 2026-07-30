import pytest
import numpy as np
from embedding_bench.backends.base import EmbeddingResult

def test_embedding_result_validation():
    # Valid dense representation
    res = EmbeddingResult(dense=np.zeros((1, 10)))
    assert res.dense is not None
    assert res.sparse is None
    assert res.multi_vector is None

    # Valid sparse representation
    res_sparse = EmbeddingResult(sparse=[{1: 0.5}])
    assert res_sparse.sparse is not None

    # Valid multi_vector representation
    res_mv = EmbeddingResult(multi_vector=[np.zeros((5, 10))])
    assert res_mv.multi_vector is not None

    # Invalid instantiation - all representations None
    with pytest.raises(ValueError, match="EmbeddingResult must contain at least one of"):
        EmbeddingResult()
