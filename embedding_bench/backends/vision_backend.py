import os
import time
import numpy as np
from PIL import Image
from .base import EmbeddingBackend, EmbeddingResult

class VisionEmbeddingBackend(EmbeddingBackend):
    def __init__(self, name, hf_model_id, dim, max_batch, device="cuda",
                 requires_image_input=True, **kwargs):
        self.name = name
        self.dim = dim
        self.max_batch = max_batch
        self.modality = "dense"
        self.device = self._resolve_device(device)
        self.requires_image_input = requires_image_input
        self._hf_model_id = hf_model_id
        
        ALLOW_MOCK = os.environ.get("EMBEDDING_BENCH_TEST_MODE") == "1"
        if ALLOW_MOCK:
            self._model = "mock"
            self._processor = "mock"
            self._device_used = "mock"
        else:
            try:
                self._model, self._processor = self._load_model(hf_model_id)
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

    def _load_model(self, hf_model_id: str):
        os.makedirs("D:/huggingface_cache", exist_ok=True)
        os.environ["HF_HOME"] = "D:/huggingface_cache"
        
        import torch
        from transformers import AutoModel, AutoProcessor
        
        processor = AutoProcessor.from_pretrained(hf_model_id, cache_dir="D:/huggingface_cache")
        # Load in half precision (FP16) on GPU if CUDA is available
        if self.device == "cuda":
            model = AutoModel.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16,
                cache_dir="D:/huggingface_cache"
            ).to(self.device)
        else:
            model = AutoModel.from_pretrained(
                hf_model_id,
                cache_dir="D:/huggingface_cache"
            ).to(self.device)
            
        model.eval()
        return model, processor

    def embed_documents(self, linked_chunks: list) -> EmbeddingResult:
        import torch
        start = time.perf_counter()
        
        if not linked_chunks:
            return EmbeddingResult(
                dense=np.empty((0, self.dim), dtype=np.float32),
                latency_ms=0.0,
                token_count=0,
                model_name=self.name
            )

        # Mock mode execution
        if self._model == "mock":
            dense_vecs = np.random.randn(len(linked_chunks), self.dim).astype(np.float32)
            latency = (time.perf_counter() - start) * 1000.0
            token_count = len(linked_chunks) * 100
            return EmbeddingResult(
                dense=dense_vecs,
                latency_ms=latency,
                token_count=token_count,
                model_name=self.name
            )

        embeddings_list = []
        token_count = 0

        # Process each chunk individually to support mixed text+image and text-only shapes safely
        for chunk in linked_chunks:
            # chunk can be a dictionary or a LinkedChunk object
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                img_path = chunk.get("figure_image_path")
            else:
                text = chunk.text
                img_path = chunk.figure_image_path
                
            image = None
            if img_path and os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    
            with torch.no_grad():
                if image is not None:
                    inputs = self._processor(text=text, images=image, return_tensors="pt")
                else:
                    inputs = self._processor(text=text, return_tensors="pt")
                    
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                token_count += int(inputs["input_ids"].numel()) if "input_ids" in inputs else len(text.split())
                
                outputs = self._model(**inputs)
                
                # Check for model embedding projection output conventions
                if hasattr(outputs, "image_embeds") and image is not None:
                    emb = outputs.image_embeds[0].cpu().numpy()
                elif hasattr(outputs, "text_embeds"):
                    emb = outputs.text_embeds[0].cpu().numpy()
                else:
                    # Fallback to mean pooling of sequence dimension
                    emb = outputs[0][0].mean(dim=0).cpu().numpy()
                    
                # Ensure correct dimension length via truncation or zero padding if mismatch occurs
                if len(emb) != self.dim:
                    if len(emb) > self.dim:
                        emb = emb[:self.dim]
                    else:
                        padded = np.zeros(self.dim, dtype=np.float32)
                        padded[:len(emb)] = emb
                        emb = padded
                        
                embeddings_list.append(emb.astype(np.float32))

        dense_vecs = np.array(embeddings_list, dtype=np.float32)
        latency = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            dense=dense_vecs,
            latency_ms=latency,
            token_count=token_count,
            model_name=self.name
        )

    def embed_query(self, text: str) -> EmbeddingResult:
        import torch
        start = time.perf_counter()

        if self._model == "mock":
            dense_vecs = np.random.randn(1, self.dim).astype(np.float32)
            latency = (time.perf_counter() - start) * 1000.0
            token_count = len(text.split()) * 2
            return EmbeddingResult(
                dense=dense_vecs,
                latency_ms=latency,
                token_count=token_count,
                model_name=self.name
            )

        with torch.no_grad():
            inputs = self._processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            token_count = int(inputs["input_ids"].numel()) if "input_ids" in inputs else len(text.split())
            
            outputs = self._model(**inputs)
            if hasattr(outputs, "text_embeds"):
                emb = outputs.text_embeds[0].cpu().numpy()
            else:
                emb = outputs[0][0].mean(dim=0).cpu().numpy()
                
            if len(emb) != self.dim:
                if len(emb) > self.dim:
                    emb = emb[:self.dim]
                else:
                    padded = np.zeros(self.dim, dtype=np.float32)
                    padded[:len(emb)] = emb
                    emb = padded

        dense_vecs = np.array([emb], dtype=np.float32)
        latency = (time.perf_counter() - start) * 1000.0
        return EmbeddingResult(
            dense=dense_vecs,
            latency_ms=latency,
            token_count=token_count,
            model_name=self.name
        )

    def cost_estimate(self, n_tokens: int) -> float:
        return 0.0

    def unload(self):
        import gc
        import torch
        if self._model and self._model != "mock":
            self._model = None
            self._processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
