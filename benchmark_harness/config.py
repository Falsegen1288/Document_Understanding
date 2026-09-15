from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class PipelineConfig:
    pipeline: Literal["baseline", "glm_ocr", "unlimited_ocr"]
    dataset: Literal["tatdqa", "unidoc"]
    chunk_strategy: str = "semantic_mesh"
    embedding_model: str = "bge-m3"
    reranking_enabled: bool = False
    top_k: int = 10
    limit: Optional[int] = None
    random_sample: bool = False
    random_seed: int = 42
    use_pal_arithmetic: bool = False
    reader_model: Optional[str] = "gemini-3.6-flash"
