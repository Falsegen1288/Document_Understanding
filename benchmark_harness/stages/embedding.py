import os
from typing import Any, List, Dict
import numpy as np

os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "D:/sentence_transformers_cache")


class EmbeddingStage:
    """Handles embedding generation for text chunks via sentence-transformers or embedding_bench registry."""

    def __init__(self, model_name: str = "bge-m3", verbose: bool = True):
        self.model_name = model_name
        self.model = None
        self.dim = 1024

        try:
            from sentence_transformers import SentenceTransformer
            hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline", "default"] else model_name
            if verbose:
                print(f"[EMBEDDING] Loading SentenceTransformer '{hf_id}'...", flush=True)
            self.model = SentenceTransformer(hf_id)
            self.dim = self.model.get_sentence_embedding_dimension() or 384
        except Exception as e:
            if verbose:
                print(f"[EMBEDDING WARNING] Failed to load embedding models ({e}). Using mock 384d dense embedding fallback...", flush=True)
            self.model = None
            self.dim = 384



    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as err:
                print(f"[EMBEDDING WARNING] Encoding failed: {err}", flush=True)

        # Fallback pseudo-random normalized embedding based on text hash
        rng_vecs = []
        for t in texts:
            seed = sum(ord(c) for c in t) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            v = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(v) or 1.0
            rng_vecs.append(v / norm)
        return np.vstack(rng_vecs)
