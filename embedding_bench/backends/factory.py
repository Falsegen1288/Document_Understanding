import yaml
from .base import EmbeddingBackend
from .local_st import LocalSTBackend
from .multi_vector import MultiVectorBackend
from .vision_backend import VisionEmbeddingBackend

class EmbeddingBackendFactory:
    _registry_cache: dict | None = None

    @classmethod
    def load_registry(cls, path: str = "embedding_bench/registry/models.yaml") -> dict:
        if cls._registry_cache is None:
            from ..config import load_models_config, REGISTRY_PATH
            cls._registry_cache = load_models_config(REGISTRY_PATH)
        return cls._registry_cache

    @classmethod
    def create(cls, model_key: str, overrides: dict | None = None) -> EmbeddingBackend:
        """
        model_key must exactly match a key in models.yaml (e.g. "bge-m3").
        overrides: optional dict to override registry values at runtime (e.g. device, quantization).
        Raises KeyError with a clear message listing valid keys if model_key is not found.
        """
        registry = cls.load_registry()
        if model_key not in registry:
            raise KeyError(f"Unknown model_key '{model_key}'. Valid keys: {list(registry.keys())}")
        cfg = {**registry[model_key], **(overrides or {})}
        backend_cls = cls._resolve_class(cfg["backend_class"])
        return backend_cls(name=model_key, **cfg)

    @staticmethod
    def _resolve_class(class_name: str):
        mapping = {
            "LocalSTBackend": LocalSTBackend,
            "MultiVectorBackend": MultiVectorBackend,
            "VisionEmbeddingBackend": VisionEmbeddingBackend,
        }
        if class_name not in mapping:
            raise ValueError(f"Unknown backend_class '{class_name}'")
        return mapping[class_name]
