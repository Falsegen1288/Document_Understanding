import os
import time
import numpy as np
from .base import EmbeddingBackend, EmbeddingResult

class LocalSTBackend(EmbeddingBackend):
    def __init__(self, name, hf_model_id, dim, max_batch, device="cuda",
                 supports_query_doc_prefix=False, query_prefix="", doc_prefix="",
                 quantization=None, revision=None, **kwargs):
        self.name = name
        self.dim = dim
        self.max_batch = max_batch
        self.supports_query_doc_prefix = supports_query_doc_prefix
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.modality = "dense"
        self.device = self._resolve_device(device)
        self._hf_model_id = hf_model_id
        self._quantization = quantization
        self.revision = revision
        
        ALLOW_MOCK = os.environ.get("EMBEDDING_BENCH_TEST_MODE") == "1"
        if ALLOW_MOCK:
            self._model = "mock"
            self._backend_type = "mock"
            self._device_used = "mock"
        else:
            try:
                self._model = self._load_model(hf_model_id, quantization)
                self._device_used = self.device
            except Exception as e:
                raise RuntimeError(
                    f"Model {hf_model_id} failed to load and EMBEDDING_BENCH_TEST_MODE is not set. "
                    f"Refusing to silently fall back to mock vectors. Original error: {e}"
                ) from e

    def _resolve_device(self, requested: str) -> str:
        import torch
        if requested == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return requested

    def _load_model(self, hf_model_id: str, quantization: str | None):
        # Redirect cache directories to Drive D
        os.makedirs("D:/huggingface_cache", exist_ok=True)
        os.makedirs("D:/sentence_transformers_cache", exist_ok=True)
        os.environ["HF_HOME"] = "D:/huggingface_cache"
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/sentence_transformers_cache"

        from sentence_transformers import SentenceTransformer
        import torch
        
        model_kwargs = {}
        if quantization:
            from ..quantization import get_quantization_config
            model_kwargs["quantization_config"] = get_quantization_config(int(quantization))
            model_kwargs["device_map"] = "auto"
            
        try:
            # Attempt loading via SentenceTransformer
            model = SentenceTransformer(
                hf_model_id,
                device=self.device if not quantization else None,
                model_kwargs=model_kwargs if quantization else None,
                trust_remote_code=True,
                cache_folder="D:/sentence_transformers_cache"
            )
            self._backend_type = "sentence_transformers"
            return model
        except Exception as e:
            print(f"Warning: Failed to load via SentenceTransformers ({e}). Falling back to raw transformers...")
            
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id, cache_dir="D:/huggingface_cache")
            if quantization:
                from ..quantization import get_quantization_config
                model = AutoModel.from_pretrained(
                    hf_model_id,
                    quantization_config=get_quantization_config(int(quantization)),
                    device_map="auto",
                    cache_dir="D:/huggingface_cache"
                )
            else:
                model = AutoModel.from_pretrained(hf_model_id, cache_dir="D:/huggingface_cache").to(self.device)
            self._backend_type = "transformers"
            self._tokenizer = tokenizer
            return model

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        import torch
        
        # Apply doc prefix if configured
        prefixed_texts = texts
        if self.supports_query_doc_prefix and self.doc_prefix:
            prefixed_texts = [self.doc_prefix + t for t in texts]

        # Mock mode check
        if self._model == "mock":
            start = time.perf_counter()
            dense_vecs = np.random.randn(len(texts), self.dim).astype(np.float32)
            latency = (time.perf_counter() - start) * 1000.0
            token_count = sum(len(t.split()) for t in texts) * 2
            return EmbeddingResult(
                dense=dense_vecs,
                latency_ms=latency,
                token_count=token_count,
                model_name=self.name,
                device_used=self._device_used
            )

        start = time.perf_counter()
        embeddings_list = []
        token_count = 0

        # Process in batches
        for i in range(0, len(prefixed_texts), self.max_batch):
            batch_texts = prefixed_texts[i:i + self.max_batch]
            
            if self._backend_type == "sentence_transformers":
                tokenizer = self._model.tokenizer
            else:
                tokenizer = self._tokenizer
                
            encoded_tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            batch_token_count = int(encoded_tokens["input_ids"].numel())
            token_count += batch_token_count
            
            if self._backend_type == "sentence_transformers":
                batch_emb = self._model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings_list.append(batch_emb)
            else:
                if not self._quantization:
                    encoded_tokens = {k: v.to(self.device) for k, v in encoded_tokens.items()}
                with torch.no_grad():
                    outputs = self._model(**encoded_tokens)
                    attention_mask = encoded_tokens["attention_mask"]
                    token_embeddings = outputs[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = input_mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    batch_emb = (sum_embeddings / sum_mask).cpu().numpy()
                    embeddings_list.append(batch_emb)

        if embeddings_list:
            dense_vecs = np.concatenate(embeddings_list, axis=0).astype(np.float32)
        else:
            dense_vecs = np.empty((0, self.dim), dtype=np.float32)

        latency = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            dense=dense_vecs,
            latency_ms=latency,
            token_count=token_count,
            model_name=self.name,
            device_used=self._device_used
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        import torch
        
        prefixed_text = text
        if self.supports_query_doc_prefix and self.query_prefix:
            prefixed_text = self.query_prefix + text

        if self._model == "mock":
            start = time.perf_counter()
            dense_vecs = np.random.randn(1, self.dim).astype(np.float32)
            latency = (time.perf_counter() - start) * 1000.0
            token_count = len(text.split()) * 2
            return EmbeddingResult(
                dense=dense_vecs,
                latency_ms=latency,
                token_count=token_count,
                model_name=self.name,
                device_used=self._device_used
            )

        start = time.perf_counter()
        
        if self._backend_type == "sentence_transformers":
            tokenizer = self._model.tokenizer
        else:
            tokenizer = self._tokenizer
            
        encoded_tokens = tokenizer([prefixed_text], padding=True, truncation=True, return_tensors="pt")
        token_count = int(encoded_tokens["input_ids"].numel())

        if self._backend_type == "sentence_transformers":
            dense_vecs = self._model.encode(
                [prefixed_text],
                convert_to_numpy=True,
                show_progress_bar=False
            )
        else:
            if not self._quantization:
                encoded_tokens = {k: v.to(self.device) for k, v in encoded_tokens.items()}
            with torch.no_grad():
                outputs = self._model(**encoded_tokens)
                attention_mask = encoded_tokens["attention_mask"]
                token_embeddings = outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                dense_vecs = (sum_embeddings / sum_mask).cpu().numpy()

        latency = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            dense=dense_vecs,
            latency_ms=latency,
            token_count=token_count,
            model_name=self.name,
            device_used=self._device_used
        )

    def cost_estimate(self, n_tokens: int) -> float:
        return 0.0

    def unload(self):
        import gc
        import torch
        if self._model and self._model != "mock":
            self._model = None
            if hasattr(self, "_tokenizer"):
                self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
