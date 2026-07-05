"""Microsoft Table Transformer (TATR) helpers."""

from __future__ import annotations

from typing import Any

from PIL import Image

_det_model: Any | None = None
_det_processor: Any | None = None
_struct_model: Any | None = None
_struct_processor: Any | None = None

DETECTION_MODEL_ID = "microsoft/table-transformer-detection"
STRUCTURE_MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"


def _load_model(model_id: str) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, AutoConfig
    import os
    import json
    import tempfile
    from huggingface_hub import hf_hub_download

    processor = AutoImageProcessor.from_pretrained(model_id)
    if hasattr(processor, "size") and processor.size is not None:
        if "shortest_edge" not in processor.size or processor.size.get("shortest_edge") is None:
            processor.size["shortest_edge"] = 800
    
    # Robust loading of the config to work around huggingface_hub validation bugs (e.g. dilation=None)
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        # If loading normally fails (due to strict dataclass type validation on HuggingFace Hub),
        # download the config manually, patch the "dilation" attribute, and load local config
        try:
            config_file = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_file, "r") as f:
                config_dict = json.load(f)
            
            if config_dict.get("dilation") is None:
                config_dict["dilation"] = False
                
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                json.dump(config_dict, tmp)
                tmp_path = tmp.name
                
            try:
                config = AutoConfig.from_pretrained(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        except Exception as inner_exc:
            # Fallback in case manual patch fails: raise the original exception
            raise RuntimeError(f"Failed to load patched config: {inner_exc}")
            
    model = AutoModelForObjectDetection.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)
    model.eval()
    return model, processor


def _detection_model() -> tuple[Any, Any]:
    global _det_model, _det_processor
    if _det_model is None or _det_processor is None:
        _det_model, _det_processor = _load_model(DETECTION_MODEL_ID)
    return _det_model, _det_processor


def _structure_model() -> tuple[Any, Any]:
    global _struct_model, _struct_processor
    if _struct_model is None or _struct_processor is None:
        _struct_model, _struct_processor = _load_model(STRUCTURE_MODEL_ID)
    return _struct_model, _struct_processor


def _run_object_detection(
    image: Image.Image,
    model: Any,
    processor: Any,
    confidence: float,
) -> list[dict[str, Any]]:
    import torch

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    result = processor.post_process_object_detection(
        outputs,
        threshold=confidence,
        target_sizes=target_sizes,
    )[0]

    detections: list[dict[str, Any]] = []
    for score, label_id, box in zip(
        result["scores"].tolist(),
        result["labels"].tolist(),
        result["boxes"].tolist(),
    ):
        label = model.config.id2label.get(int(label_id), f"class_{int(label_id)}")
        detections.append(
            {
                "label": label,
                "confidence": round(float(score), 4),
                "bbox": [round(float(value), 2) for value in box],
            }
        )
    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return detections


def detect_tables(image: Image.Image, confidence: float = 0.7) -> list[dict[str, Any]]:
    """Detect table boxes in a full page image."""
    try:
        model, processor = _detection_model()
        return _run_object_detection(image, model, processor, confidence)
    except Exception as exc:  # noqa: BLE001
        return [{"engine": "TATR", "error": str(exc)}]


def extract_table_structure(
    table_image: Image.Image,
    confidence: float = 0.5,
) -> dict[str, Any]:
    """Detect rows, columns, headers, and spanning cells in a table crop."""
    try:
        model, processor = _structure_model()
        detections = _run_object_detection(table_image, model, processor, confidence)
    except Exception as exc:  # noqa: BLE001
        return {"engine": "TATR", "error": str(exc)}

    rows = []
    columns = []
    cells = []
    for detection in detections:
        label = detection["label"].lower()
        if "row" in label and "header" not in label:
            rows.append(detection)
        elif "column" in label and "header" not in label:
            columns.append(detection)
        else:
            cells.append(detection)
    return {"rows": rows, "columns": columns, "cells": cells, "engine": "TATR"}


def extract_tables(
    image: Image.Image,
    det_confidence: float = 0.7,
    struct_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect tables in a page and run structure recognition on each crop."""
    detections = detect_tables(image, confidence=det_confidence)
    if detections and "error" in detections[0]:
        return detections

    tables: list[dict[str, Any]] = []
    page = image.convert("RGB")
    width, height = page.size
    for index, detection in enumerate(detections, start=1):
        x0, y0, x1, y1 = detection["bbox"]
        crop = page.crop(
            (
                max(0, int(x0)),
                max(0, int(y0)),
                min(width, int(x1)),
                min(height, int(y1)),
            )
        )
        tables.append(
            {
                "table_index": index,
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "structure": extract_table_structure(crop, confidence=struct_confidence),
                "engine": "TATR",
            }
        )
    return tables


run_tatr = extract_tables
