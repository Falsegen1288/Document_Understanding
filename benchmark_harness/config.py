from dataclasses import dataclass, field
from typing import Literal, Optional, Any


@dataclass
class PipelineConfig:
    pipeline: str = "baseline"  # "baseline", "glm_ocr", "unlimited_ocr", or "custom"
    dataset: str = "unidoc"     # "unidoc", "tatdqa", or "custom"
    dataset_file: Optional[str] = None  # Path to arbitrary user-provided benchmark JSON

    # Custom component options
    layout: str = "doclayout_yolo"
    ocr: str = "paddleocr"
    table: str = "docling_tableformer"
    figures: str = "gemini"
    chunk_strategy: str = "semantic_mesh"
    embedding_model: str = "bge-m3"
    retrieval: str = "rrf_hybrid"
    reader_model: Optional[str] = "gemini-3.6-flash"
    use_pal_arithmetic: bool = False

    # Execution controls
    reranking_enabled: bool = False
    top_k: int = 10
    limit: Optional[int] = None
    random_sample: bool = False
    random_seed: int = 42
