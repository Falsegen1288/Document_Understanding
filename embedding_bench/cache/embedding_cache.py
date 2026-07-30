import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any

class EmbeddingCache:
    def __init__(self, model_key: str, cache_root: str = "outputs/.embeddings_cache"):
        self.model_key = model_key
        self.cache_dir = Path(cache_root) / model_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, text: str, doc_prefix: str = "") -> str:
        data_to_hash = f"{text}|{self.model_key}|{doc_prefix}"
        return hashlib.sha256(data_to_hash.encode("utf-8")).hexdigest()

    def get(self, text: str, doc_prefix: str = "") -> Optional[Any]:
        """Returns cached embedding (np.ndarray or dict/EmbeddingResult) or None."""
        key = self._get_key(text, doc_prefix)
        file_path = self.cache_dir / f"{key}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    def get_batch(self, texts: list[str], doc_prefix: str = "") -> tuple[list[Any], list[int]]:
        """
        Returns (results, missing_indices).
        results[i] is the cached embedding for texts[i], or None if missing.
        missing_indices is the list of indices in texts that need to be computed.
        """
        results = []
        missing_indices = []
        for idx, text in enumerate(texts):
            res = self.get(text, doc_prefix)
            results.append(res)
            if res is None:
                missing_indices.append(idx)
        return results, missing_indices

    def put(self, text: str, embedding: Any, doc_prefix: str = "") -> None:
        key = self._get_key(text, doc_prefix)
        file_path = self.cache_dir / f"{key}.pkl"
        try:
            with open(file_path, "wb") as f:
                pickle.dump(embedding, f)
        except Exception:
            pass

    def put_batch(self, texts: list[str], embeddings: list[Any], doc_prefix: str = "") -> None:
        for text, emb in zip(texts, embeddings):
            self.put(text, emb, doc_prefix)

    @classmethod
    def clear(cls, model_key: str, cache_root: str = "outputs/.embeddings_cache") -> None:
        """Wipes only that model's cache directory."""
        cache_dir = Path(cache_root) / model_key
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
