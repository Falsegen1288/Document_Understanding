import os
from typing import List, Tuple, Optional
import torch

os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "D:/sentence_transformers_cache")
os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/huggingface_cache")

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder for (query, passage) -> relevance_score reranking."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            print(f"[RERANKER] Loading CrossEncoder model '{model_name}' on device '{self.device}'...", flush=True)
            self.model = CrossEncoder(model_name, device=self.device)
            self.model_name = model_name
        except Exception as e:
            fallback = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            print(f"[RERANKER WARNING] Failed to load '{model_name}' ({e}). Falling back to '{fallback}'...", flush=True)
            self.model = CrossEncoder(fallback, device=self.device)
            self.model_name = fallback

    def rerank(self, query: str, passages: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Returns list of (original_index, score) sorted descending by score.
        """
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self.model.predict(pairs)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[:top_k] if top_k else ranked
