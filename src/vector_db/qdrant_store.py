"""
Qdrant Vector Store (Embedded Mode).
------------------------------------
Provides persistent, on-disk vector storage for multimodal document chunks.
- Runs embedded inside Python with zero external server/daemon setup.
- Stores dense 1024d vectors (BGE-M3) with Cosine similarity HNSW indexing.
- Preserves full rich metadata payload (lineage, table_pointers, cross_references).
- Fast point retrieval by deterministic UUID for graph traversal.
"""
from __future__ import annotations

import os
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = os.path.join(".cache", "vectordb", "qdrant")
DEFAULT_COLLECTION_NAME = "document_chunks"
DEFAULT_VECTOR_DIM = 1024


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Converts an arbitrary string chunk_id into a deterministic UUID string for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)))


class QdrantVectorStore:
    def __init__(
        self,
        storage_path: str = DEFAULT_STORAGE_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_dim: int = DEFAULT_VECTOR_DIM,
        distance: models.Distance = models.Distance.COSINE,
        recreate_collection: bool = False,
    ):
        self.storage_path = storage_path
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.distance = distance

        os.makedirs(self.storage_path, exist_ok=True)
        self.client = QdrantClient(path=self.storage_path)

        # Initialize collection if not present
        self._init_collection(recreate=recreate_collection)

    def _init_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=self.distance,
                    on_disk=True,
                ),
            )
            logger.info(f"[Qdrant] Created collection '{self.collection_name}' (dim={self.vector_dim}, distance={self.distance}) at '{self.storage_path}'.")
        else:
            logger.info(f"[Qdrant] Connected to existing collection '{self.collection_name}' at '{self.storage_path}'.")

    def upsert_chunks(
        self,
        chunks: List[Union[Dict[str, Any], Any]],
        embeddings: Union[np.ndarray, List[List[float]]],
    ) -> int:
        """
        Upserts chunk records along with their dense vectors into Qdrant.
        Stores full rich metadata payload (lineage, table_pointers, cross_references).
        """
        if not chunks:
            return 0

        if isinstance(embeddings, np.ndarray):
            vec_list = embeddings.tolist()
        else:
            vec_list = embeddings

        if len(chunks) != len(vec_list):
            raise ValueError(f"Number of chunks ({len(chunks)}) does not match number of embeddings ({len(vec_list)})")

        points = []
        for idx, item in enumerate(chunks):
            chunk_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)

            raw_cid = str(chunk_dict.get("chunk_id", f"chunk_{idx}"))
            pt_id = chunk_id_to_uuid(raw_cid)
            vec = [float(v) for v in vec_list[idx]]

            payload = {
                "chunk_id": raw_cid,
                "doc_id": chunk_dict.get("doc_id", chunk_dict.get("doc_filename", "unknown")),
                "doc_filename": chunk_dict.get("doc_filename", "unknown"),
                "page": chunk_dict.get("page", 1),
                "page_span": chunk_dict.get("page_span", [chunk_dict.get("page", 1)]),
                "element_types": chunk_dict.get("element_types", ["text"]),
                "text": chunk_dict.get("text", ""),
                "token_count": chunk_dict.get("token_count", 0),
                "parent_section": chunk_dict.get("parent_section"),
                "lineage": chunk_dict.get("lineage", {}),
                "table_pointers": chunk_dict.get("table_pointers", {}),
                "cross_references": chunk_dict.get("cross_references", {}),
                "metadata": chunk_dict.get("metadata", {}),
            }

            points.append(models.PointStruct(
                id=pt_id,
                vector=vec,
                payload=payload,
            ))

        # Batch upload in chunks of 100
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            total_upserted += len(batch)

        logger.info(f"[Qdrant] Successfully upserted {total_upserted} chunks into '{self.collection_name}'.")
        return total_upserted

    def search_dense(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 10,
        query_filter: Optional[models.Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes HNSW nearest-neighbor vector search.
        Returns list of matched dicts: {"chunk_id": ..., "score": ..., "text": ..., "payload": ...}
        """
        if isinstance(query_vector, np.ndarray):
            q_vec = query_vector.flatten().tolist()
        else:
            q_vec = [float(v) for v in query_vector]

        # Use query_points (modern API in qdrant-client >= 1.9)
        search_res = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results = []
        for pt in search_res.points:
            payload = pt.payload or {}
            results.append({
                "chunk_id": payload.get("chunk_id", str(pt.id)),
                "point_id": pt.id,
                "score": float(pt.score) if pt.score is not None else 0.0,
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id"),
                "payload": payload,
            })
        return results

    def get_by_chunk_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Directly fetches multiple points by their string chunk_ids using deterministic UUIDs.
        Used for instant relational graph traversal (summary rows, citations, neighboring narrative).
        """
        if not chunk_ids:
            return {}

        pt_ids = [chunk_id_to_uuid(cid) for cid in chunk_ids]
        retrieved = self.client.retrieve(
            collection_name=self.collection_name,
            ids=pt_ids,
            with_payload=True,
        )

        lookup = {}
        for pt in retrieved:
            payload = pt.payload or {}
            cid = payload.get("chunk_id", str(pt.id))
            lookup[cid] = {
                "chunk_id": cid,
                "point_id": pt.id,
                "text": payload.get("text", ""),
                "doc_id": payload.get("doc_id"),
                "payload": payload,
            }
        return lookup

    def count(self) -> int:
        """Returns total points in the collection."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    def clear(self) -> None:
        """Clears all vectors in the collection."""
        self._init_collection(recreate=True)

    def close(self) -> None:
        """Explicitly closes client and releases locks."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass
