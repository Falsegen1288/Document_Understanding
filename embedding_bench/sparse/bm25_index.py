import pickle
import re
from pathlib import Path

class BM25Index:
    def __init__(self, corpus_chunks: list):
        self.chunk_ids = [c["chunk_id"] for c in corpus_chunks]
        self.corpus = corpus_chunks
        
        tokenized_corpus = [self._tokenize(c.get("text", "")) for c in corpus_chunks]
        
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        # Simple whitespace + lowercase + punctuation-strip tokenizer
        # Preserves alphanumeric digit/letter codes (like 181800)
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        results = list(zip(self.chunk_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @classmethod
    def load_or_build(cls, corpus_chunks: list, cache_path: Path, force_recompute: bool = False):
        if cache_path.exists() and not force_recompute:
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        index = cls(corpus_chunks)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(index, f)
        return index
