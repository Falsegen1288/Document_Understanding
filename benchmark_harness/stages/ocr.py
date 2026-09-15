import os
from pathlib import Path
from typing import Dict, Any, Optional
from algorithms.text_extraction.glm_ocr.extractor import GLMOCRExtractor
from algorithms.table_extraction.glm_ocr.extractor import GLMOCRTableExtractor
from algorithms.image_extraction.glm_ocr.extractor import GLMOCRFigureExtractor

TATDQA_PDF_ROOT = Path("data/tatdqa/source_pdfs")


def sanity_check_pdf_corpus(expected_count: int = 274) -> None:
    """Run once after setting TATDQA_PDF_ROOT — confirms the corpus is
    actually where you think it is before kicking off a full re-benchmark."""
    found = list(TATDQA_PDF_ROOT.glob("*.pdf"))
    print(f"Found {len(found)} PDFs at {TATDQA_PDF_ROOT} (expected {expected_count})")
    if len(found) != expected_count:
        print("⚠️  Count mismatch — verify TATDQA_PDF_ROOT before running the full corpus.")


_PIPELINE_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_pipeline_backend(pipeline: str) -> Dict[str, Any]:
    if pipeline in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[pipeline].copy()

    if pipeline == "glm_ocr":
        text_ext = GLMOCRExtractor()
        table_ext = GLMOCRTableExtractor()
        image_ext = GLMOCRFigureExtractor()
        degraded = text_ext.is_degraded or table_ext.is_degraded or image_ext.is_degraded
        res = {
            "skip_ocr": False,
            "pipeline": pipeline,
            "mode": "glm_ocr" if not degraded else "baseline_degraded",
            "text_extractor": text_ext,
            "table_extractor": table_ext,
            "image_extractor": image_ext,
            "degraded": degraded
        }
        _PIPELINE_CACHE[pipeline] = res
        return res.copy()

    elif pipeline == "unlimited_ocr":
        return {
            "skip_ocr": False,
            "pipeline": pipeline,
            "mode": "unlimited_ocr",
            "force_ocr_all_pages": True
        }

    else:
        # Default Baseline
        return {
            "skip_ocr": False,
            "pipeline": pipeline,
            "mode": "baseline",
            "force_ocr_all_pages": False
        }


def get_ocr_backend(pipeline: str, dataset: str, doc: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns OCR configuration and backend handlers for the target pipeline and dataset.
    TAT-DQA is routed through its real source PDF when available, matching unidoc ingestion.
    """
    backend = _get_pipeline_backend(pipeline)

    if dataset == "tatdqa" and doc is not None and "doc" in doc:
        source_pdf = doc["doc"].get("source")
        page_num = doc["doc"].get("page")
        pdf_path = TATDQA_PDF_ROOT / source_pdf if source_pdf else None
        pdf_exists = pdf_path.exists() if pdf_path else False

        backend.update({
            "input_pdf_path": str(pdf_path) if pdf_exists else None,
            "page_number": page_num,
            "source_pdf": source_pdf,
            "has_backing_pdf": pdf_exists,
            "note": "TAT-DQA routed through real source PDF when available." if pdf_exists else "TAT-DQA source PDF missing, using facts fallback."
        })

    return backend
