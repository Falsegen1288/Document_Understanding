from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

@dataclass
class EmbeddingResult:
    dense: Optional[np.ndarray] = None          # shape (n_texts, dim), float32
    sparse: Optional[list[dict[int, float]]] = None  # one dict per text: token_id -> weight
    multi_vector: Optional[list[np.ndarray]] = None   # one array per text: (n_tokens, dim)
    latency_ms: float = 0.0
    token_count: int = 0
    model_name: str = ""
    device_used: str = "cpu"

    def __post_init__(self):
        # Validate at least one representation is present
        if self.dense is None and self.sparse is None and self.multi_vector is None:
            raise ValueError("EmbeddingResult must contain at least one of dense/sparse/multi_vector")


class EmbeddingBackend(ABC):
    """
    Every embedding model integration (local or API) MUST subclass this.
    Do not add model-specific methods to this base class — put them in the subclass.
    """

    name: str
    modality: Literal["dense", "dense+sparse", "multi-vector"]
    dim: int
    max_batch: int
    supports_query_doc_prefix: bool
    query_prefix: str = ""
    doc_prefix: str = ""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of document/chunk texts. MUST internally batch to self.max_batch."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query. Apply self.query_prefix if supports_query_doc_prefix is True."""
        pass

    @abstractmethod
    def cost_estimate(self, n_tokens: int) -> float:
        """Return USD cost for n_tokens. Return 0.0 for local models."""
        pass

    def health_check(self) -> bool:
        """Default implementation: embed a single test string and confirm no exception."""
        try:
            result = self.embed_query("health check test string")
            return result is not None
        except Exception:
            return False
