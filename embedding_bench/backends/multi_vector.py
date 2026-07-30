import os
import time
import numpy as np
from .local_st import LocalSTBackend
from .base import EmbeddingResult

class MultiVectorBackend(LocalSTBackend):
    """
    BGE-M3 uniquely returns dense + sparse + multi-vector (ColBERT-style) from
    ONE forward pass. Use the FlagEmbedding library's BGEM3FlagModel, NOT plain
    sentence-transformers, to get all three outputs without three separate calls.
    """
    def __init__(self, name, hf_model_id, dim, max_batch, device="cuda",
                 supports_query_doc_prefix=False, query_prefix="", doc_prefix="",
                 quantization=None, **kwargs):
        super().__init__(name, hf_model_id, dim, max_batch, device,
                         supports_query_doc_prefix, query_prefix, doc_prefix,
                         quantization, **kwargs)
        self.modality = "dense+sparse"

    def _load_model(self, hf_model_id: str, quantization: str | None):
        os.makedirs("D:/huggingface_cache", exist_ok=True)
        os.environ["HF_HOME"] = "D:/huggingface_cache"
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel(hf_model_id, use_fp16=True if self.device == "cuda" else False, device=self.device)


    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        if self._model == "mock":
            return super().embed_documents(texts)

        start = time.perf_counter()
        if not texts:
            return EmbeddingResult(
                dense=np.empty((0, self.dim), dtype=np.float32),
                sparse=[],
                multi_vector=[],
                latency_ms=0.0,
                token_count=0,
                model_name=self.name
            )

        encoded = self._model.encode(
            texts,
            batch_size=self.max_batch,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        dense_vecs = encoded["dense_vecs"].astype(np.float32)
        
        sparse_vecs = []
        for weight_dict in encoded["lexical_weights"]:
            new_dict = {}
            for k, v in weight_dict.items():
                try:
                    new_dict[int(k)] = float(v)
                except ValueError:
                    new_dict[k] = float(v)
            sparse_vecs.append(new_dict)

        multi_vecs = [vec.astype(np.float32) for vec in encoded["colbert_vecs"]]

        token_count = 0
        if hasattr(self._model, "tokenizer"):
            encoded_tokens = self._model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            token_count = int(encoded_tokens["input_ids"].numel())
        else:
            token_count = sum(len(t.split()) for t in texts) * 2

        latency = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            dense=dense_vecs,
            sparse=sparse_vecs,
            multi_vector=multi_vecs,
            latency_ms=latency,
            token_count=token_count,
            model_name=self.name,
            device_used=self._device_used
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        if self._model == "mock":
            return super().embed_query(text)

        start = time.perf_counter()
        encoded = self._model.encode(
            [text],
            batch_size=1,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        dense_vecs = encoded["dense_vecs"].astype(np.float32)
        
        sparse_vecs = []
        for weight_dict in encoded["lexical_weights"]:
            new_dict = {}
            for k, v in weight_dict.items():
                try:
                    new_dict[int(k)] = float(v)
                except ValueError:
                    new_dict[k] = float(v)
            sparse_vecs.append(new_dict)

        multi_vecs = [vec.astype(np.float32) for vec in encoded["colbert_vecs"]]

        token_count = 0
        if hasattr(self._model, "tokenizer"):
            encoded_tokens = self._model.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            token_count = int(encoded_tokens["input_ids"].numel())
        else:
            token_count = len(text.split()) * 2

        latency = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            dense=dense_vecs,
            sparse=sparse_vecs,
            multi_vector=multi_vecs,
            latency_ms=latency,
            token_count=token_count,
            model_name=self.name,
            device_used=self._device_used
        )
