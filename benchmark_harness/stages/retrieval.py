"""
RRF Hybrid Retriever Stage with Qdrant Vector Store Integration.
---------------------------------------------------------------
1. Stores dense embeddings and full rich metadata in persistent Qdrant HNSW index.
2. Fuses dense vector similarity search with sparse Okapi BM25 token matching via RRF (k=60).
3. Relational Context Expansion ("Small-to-Big"):
   - Expands retrieved table row chunks with their total/summary rows (summary_row_id).
   - Expands text chunks citing charts/tables with target figure/table transcriptions.
   - Expands cut narratives with next_chunk_id / prev_chunk_id.
"""
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np

from src.table_indexing.tokenizer import TableEntityTokenizer
from embedding_bench.sparse.fusion import reciprocal_rank_fusion
from src.routing.dispatch_router import DispatchRouter
from src.vector_db.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class RRFHybridRetrieverStage:
    def __init__(
        self,
        embedding_stage: Any,
        k_rrf: int = 60,
        use_qdrant: bool = True,
        qdrant_path: str = ".cache/vectordb/qdrant",
        collection_name: str = "document_chunks",
    ):
        self.embedding_stage = embedding_stage
        self.k_rrf = k_rrf
        self.use_qdrant = use_qdrant
        self.chunks = []
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {}
        self.doc_ids = []
        self.doc_tokens = {}
        self.doc_freqs = defaultdict(int)
        self.dense_embeddings = None

        self.qdrant_store: Optional[QdrantVectorStore] = None
        if self.use_qdrant:
            try:
                self.qdrant_store = QdrantVectorStore(
                    storage_path=qdrant_path,
                    collection_name=collection_name,
                    vector_dim=self.embedding_stage.dim,
                    recreate_collection=True,  # Fresh collection per run
                )
            except Exception as e:
                logger.warning(f"[RETRIEVER] Failed to initialize QdrantVectorStore: {e}. Falling back to in-memory NumPy.")
                self.qdrant_store = None

    def index_chunks(self, chunk_records: List[Dict[str, Any]]):
        """
        Indexes chunks into both Qdrant HNSW dense store and Sparse BM25 inverted index.
        """
        self.chunks = chunk_records
        self.doc_ids = [c["chunk_id"] for c in chunk_records]
        self.chunk_lookup = {c["chunk_id"]: c for c in chunk_records}
        texts = [c.get("text", "") for c in chunk_records]

        # 1. Sparse BM25 setup
        self.doc_tokens = {}
        self.doc_freqs = defaultdict(int)
        for c in chunk_records:
            cid = c["chunk_id"]
            toks = TableEntityTokenizer.tokenize(c.get("text", ""))
            self.doc_tokens[cid] = toks
            for t in set(toks):
                self.doc_freqs[t] += 1

        # 2. Dense Embeddings generation
        if texts:
            self.dense_embeddings = self.embedding_stage.encode(texts)
        else:
            self.dense_embeddings = np.zeros((0, self.embedding_stage.dim), dtype=np.float32)

        # 3. Ingest into Qdrant Persistent Store
        if self.qdrant_store is not None and len(chunk_records) > 0:
            try:
                self.qdrant_store.upsert_chunks(chunk_records, self.dense_embeddings)
            except Exception as e:
                logger.warning(f"[RETRIEVER] Qdrant upsert failed: {e}. In-memory embeddings remain active.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Executes hybrid retrieval: Qdrant Dense HNSW + Sparse BM25 fused with RRF.
        Returns list of tuples: [(chunk_id, fused_rrf_score), ...]
        """
        if not self.chunks:
            return []

        # Check dispatch query routing (table vs prose)
        q_intent = DispatchRouter.classify_query(query)

        # 1. Dense Search
        q_emb = self.embedding_stage.encode([query])
        dense_ranked = []

        if self.qdrant_store is not None:
            try:
                qdrant_matches = self.qdrant_store.search_dense(q_emb, top_k=top_k * 3)
                dense_ranked = [(m["chunk_id"], m["score"]) for m in qdrant_matches]
            except Exception as e:
                logger.debug(f"[RETRIEVER] Qdrant search fallback: {e}")
                dense_ranked = []

        if not dense_ranked:
            dense_scores = np.dot(self.dense_embeddings, q_emb.T).squeeze() if len(self.chunks) > 0 else np.array([])
            if dense_scores.ndim == 0:
                dense_scores = np.array([float(dense_scores)])
            dense_ranked = [
                (self.doc_ids[i], float(dense_scores[i]))
                for i in np.argsort(-dense_scores)
            ]

        # 2. Sparse BM25 Scores
        q_tokens = TableEntityTokenizer.tokenize(query)
        bm25_scores = {}
        total_docs = len(self.chunks)
        avg_len = sum(len(toks) for toks in self.doc_tokens.values()) / max(1, total_docs)
        k1, b = 1.5, 0.75

        for cid, doc_toks in self.doc_tokens.items():
            doc_len = len(doc_toks)
            tok_counts = defaultdict(int)
            for t in doc_toks:
                tok_counts[t] += 1
            score = 0.0
            for qt in q_tokens:
                if qt in tok_counts:
                    df = self.doc_freqs[qt]
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
                    tf = tok_counts[qt]
                    denom = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
                    score += idf * (tf * (k1 + 1.0)) / denom
            bm25_scores[cid] = score

        sparse_ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])

        # 3. RRF Fusion (k=60)
        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=self.k_rrf)
        return fused[:top_k]

    def expand_context(
        self,
        retrieved_cids: List[str],
        max_total_chunks: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Relational Context Expansion ("Small-to-Big"):
        For the top retrieved chunk IDs, follows their relational pointers:
        - Table rows: pulls summary_row_id and parent schema
        - Text citations: pulls referenced figure/table chunks
        - Narrative flow: pulls next_chunk_id / prev_chunk_id
        Returns an ordered list of unique chunk dictionaries.
        """
        expanded_result: List[Dict[str, Any]] = []
        seen_cids = set()

        # Phase 1: Add primary retrieved chunks in rank order
        for cid in retrieved_cids:
            chunk = self._get_chunk(cid)
            if chunk and cid not in seen_cids:
                seen_cids.add(cid)
                expanded_result.append(chunk)

        # Phase 2: Traverse relational pointers for each retrieved chunk
        graph_neighbors_to_fetch = []
        for chunk in list(expanded_result):
            # 1. Table Pointers: Summary/Totals Row
            t_ptrs = chunk.get("table_pointers") or {}
            if isinstance(t_ptrs, dict):
                sum_row_id = t_ptrs.get("summary_row_id")
                if sum_row_id and sum_row_id not in seen_cids:
                    graph_neighbors_to_fetch.append((sum_row_id, "table_summary_row"))

            # 2. Cross-Reference Pointers: Figures & Tables cited in text
            xrefs = chunk.get("cross_references") or {}
            if isinstance(xrefs, dict):
                for fig_id in xrefs.get("figures", []):
                    if fig_id not in seen_cids:
                        graph_neighbors_to_fetch.append((fig_id, "cited_figure"))
                for tbl_id in xrefs.get("tables", []):
                    if tbl_id not in seen_cids:
                        graph_neighbors_to_fetch.append((tbl_id, "cited_table"))

            # 3. Lineage Pointers: Narrative next/prev
            lineage = chunk.get("lineage") or {}
            if isinstance(lineage, dict):
                next_cid = lineage.get("next_chunk_id")
                if next_cid and next_cid not in seen_cids:
                    graph_neighbors_to_fetch.append((next_cid, "narrative_continuation"))

        # Phase 3: Fetch and append graph neighbors up to budget
        for neighbor_cid, role in graph_neighbors_to_fetch:
            if len(expanded_result) >= max_total_chunks:
                break
            if neighbor_cid in seen_cids:
                continue

            neighbor_chunk = self._get_chunk(neighbor_cid)
            if neighbor_chunk:
                seen_cids.add(neighbor_cid)
                # Annotate that this chunk was retrieved via graph expansion
                annotated_chunk = dict(neighbor_chunk)
                annotated_chunk["retrieval_expansion_role"] = role
                expanded_result.append(annotated_chunk)

        return expanded_result

    def _get_chunk(self, cid: str) -> Optional[Dict[str, Any]]:
        """Fast lookup by chunk_id using in-memory index or Qdrant."""
        if cid in self.chunk_lookup:
            return self.chunk_lookup[cid]

        if self.qdrant_store is not None:
            pts = self.qdrant_store.get_by_chunk_ids([cid])
            if cid in pts:
                payload = pts[cid].get("payload", {})
                return {
                    "chunk_id": cid,
                    "text": pts[cid].get("text", payload.get("text", "")),
                    "doc_id": pts[cid].get("doc_id", payload.get("doc_id")),
                    "table_pointers": payload.get("table_pointers", {}),
                    "cross_references": payload.get("cross_references", {}),
                    "lineage": payload.get("lineage", {}),
                    "page_span": payload.get("page_span", []),
                }
        return None
