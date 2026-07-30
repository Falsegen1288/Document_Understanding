"""Embedding utilities shared across strategies (Phase 4+) and intrinsic metrics
(Phase 6). Wraps a local sentence-transformers model by default; falls back or
switches to an API-based embedder if configured."""
import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"  # fast, 384-dim, good enough for coherence scoring


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalSentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """Singleton accessor so we don't reload the model per-chunker-call.
    Later phases should call this rather than instantiating embedders directly."""
    logger.info(f"Loading local embedder: {DEFAULT_LOCAL_MODEL}")
    return LocalSentenceTransformerEmbedder()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors. Returns 0.0 if either vector
    is all-zero (undefined cosine) instead of raising."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """N x N cosine similarity matrix for a batch of embeddings. Used by intrinsic
    coherence metrics in Phase 6."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8  # avoid div-by-zero for empty-text embeddings
    normalized = embeddings / norms
    return normalized @ normalized.T
