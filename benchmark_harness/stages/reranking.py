from typing import List, Tuple
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


class RerankingStage:
    """Reranking Stage: Wraps CrossEncoderReranker to re-score top-N candidates."""

    def __init__(self, enabled: bool = False, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.enabled = enabled
        self.reranker = None
        if enabled:
            try:
                self.reranker = CrossEncoderReranker(model_name=model_name)
            except Exception as e:
                print(f"[RERANKING WARNING] Failed to initialize reranker ({e}). Reranking disabled.", flush=True)
                self.enabled = False

    def rerank(self, query: str, candidate_chunks: List[Tuple[str, str]], top_k: int = 10) -> List[Tuple[str, str]]:
        """
        candidate_chunks: [(chunk_id, chunk_text)]
        Returns reranked [(chunk_id, chunk_text)] list of length <= top_k.
        """
        if not candidate_chunks:
            return []

        if not self.enabled or self.reranker is None:
            return candidate_chunks[:top_k]

        passages = [c[1] for c in candidate_chunks]
        ranked_indices_scores = self.reranker.rerank(query, passages, top_k=top_k)

        reranked_chunks = []
        for idx, score in ranked_indices_scores:
            if idx < len(candidate_chunks):
                reranked_chunks.append(candidate_chunks[idx])

        return reranked_chunks
