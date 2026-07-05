"""
TATR Table Transformer Extractor
---------------------------------
Uses Microsoft's Table Transformer models for:
  1. Table detection (locating tables in page images)
  2. Table structure recognition (identifying rows, columns, cells)
  3. Cell content extraction via text mapping

Models:
  - Detection: microsoft/table-transformer-detection
  - Structure: microsoft/table-transformer-structure-recognition-v1.1-all
"""

import os
from PIL import Image
import pandas as pd

_det_model = None
_det_processor = None
_str_model = None
_str_processor = None


def _load_detection_model():
    """Lazy-load the table detection model."""
    global _det_model, _det_processor
    if _det_model is None:
        from transformers import AutoModelForObjectDetection, AutoImageProcessor
        model_name = "microsoft/table-transformer-detection"
        print(f"[INFO] Loading TATR detection model: {model_name}")
        _det_processor = AutoImageProcessor.from_pretrained(model_name)
        if hasattr(_det_processor, "size") and _det_processor.size is not None:
            if "shortest_edge" not in _det_processor.size or _det_processor.size.get("shortest_edge") is None:
                _det_processor.size["shortest_edge"] = 800
        _det_model = AutoModelForObjectDetection.from_pretrained(model_name)
        _det_model.eval()
        print("[INFO] TATR detection model loaded.")
    return _det_model, _det_processor


def _load_structure_model():
    """Lazy-load the table structure recognition model."""
    global _str_model, _str_processor
    if _str_model is None:
        from transformers import AutoModelForObjectDetection, AutoImageProcessor, AutoConfig
        import os
        import json
        import tempfile
        from huggingface_hub import hf_hub_download
        
        model_name = "microsoft/table-transformer-structure-recognition-v1.1-all"
        print(f"[INFO] Loading TATR structure model: {model_name}")
        _str_processor = AutoImageProcessor.from_pretrained(model_name)
        if hasattr(_str_processor, "size") and _str_processor.size is not None:
            if "shortest_edge" not in _str_processor.size or _str_processor.size.get("shortest_edge") is None:
                _str_processor.size["shortest_edge"] = 800
        
        try:
            config = AutoConfig.from_pretrained(model_name)
        except Exception:
            try:
                config_file = hf_hub_download(repo_id=model_name, filename="config.json")
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
                raise RuntimeError(f"Failed to load patched config: {inner_exc}")
                
        _str_model = AutoModelForObjectDetection.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        _str_model.eval()
        print("[INFO] TATR structure model loaded.")
    return _str_model, _str_processor


def _detect_tables_in_image(image: Image.Image, confidence: float = 0.7) -> list[dict]:
    """
    Detect table bounding boxes in a page image.
    
    Returns list of dicts with 'bbox' [x0,y0,x1,y1] and 'confidence'.
    """
    import torch
    model, processor = _load_detection_model()
    
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process: convert to pixel coordinates
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, threshold=confidence, target_sizes=target_sizes)[0]
    
    tables = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        if label_name in ("table", "table rotated"):
            tables.append({
                "bbox": box.tolist(),  # [x0, y0, x1, y1]
                "confidence": score.item(),
                "label": label_name
            })
    return tables


def _recognize_structure(table_image: Image.Image, confidence: float = 0.5) -> dict:
    """
    Recognize table structure (rows, columns, cells) from a cropped table image.
    
    Returns dict with 'rows', 'columns', 'cells' — each a list of bbox dicts.
    """
    import torch
    model, processor = _load_structure_model()
    
    inputs = processor(images=table_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([table_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=confidence, target_sizes=target_sizes)[0]
    
    structure = {"rows": [], "columns": [], "cells": [], "headers": []}
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        entry = {"bbox": box.tolist(), "confidence": score.item()}
        
        if label_name == "table row":
            structure["rows"].append(entry)
        elif label_name == "table column":
            structure["columns"].append(entry)
        elif label_name in ("table spanning cell", "table projected row header"):
            structure["cells"].append(entry)
        elif label_name == "table column header":
            structure["headers"].append(entry)
    
    # Sort rows top-to-bottom, columns left-to-right
    structure["rows"].sort(key=lambda r: r["bbox"][1])
    structure["columns"].sort(key=lambda c: c["bbox"][0])
    
    return structure


def _build_grid(structure: dict, table_image: Image.Image, 
                pdf_page=None, table_bbox=None) -> pd.DataFrame:
    """
    Build a DataFrame from detected rows and columns by extracting text
    from each cell intersection.
    
    Args:
        structure: Output from _recognize_structure
        table_image: Cropped table PIL Image
        pdf_page: Optional fitz page for native text extraction
        table_bbox: Table bbox in PDF coordinates [x0,y0,x1,y1] if pdf_page is provided
    """
    rows = structure["rows"]
    cols = structure["columns"]
    
    if not rows or not cols:
        return pd.DataFrame()
    
    n_rows = len(rows)
    n_cols = len(cols)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    
    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            # Cell bbox = intersection of row and column
            cell_x0 = max(row["bbox"][0], col["bbox"][0])
            cell_y0 = max(row["bbox"][1], col["bbox"][1])
            cell_x1 = min(row["bbox"][2], col["bbox"][2])
            cell_y1 = min(row["bbox"][3], col["bbox"][3])
            
            if cell_x1 <= cell_x0 or cell_y1 <= cell_y0:
                continue
            
            # Try native PDF text extraction first (more accurate)
            cell_text = ""
            if pdf_page is not None and table_bbox is not None:
                import fitz
                # Map cell coords back to PDF page coordinates
                tw = table_bbox[2] - table_bbox[0]
                th = table_bbox[3] - table_bbox[1]
                iw, ih = table_image.size
                
                pdf_x0 = table_bbox[0] + (cell_x0 / iw) * tw
                pdf_y0 = table_bbox[1] + (cell_y0 / ih) * th
                pdf_x1 = table_bbox[0] + (cell_x1 / iw) * tw
                pdf_y1 = table_bbox[1] + (cell_y1 / ih) * th
                
                rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
                cell_text = pdf_page.get_text("text", clip=rect).strip()
            
            # Fallback: OCR the cell region
            if not cell_text:
                try:
                    cell_crop = table_image.crop((cell_x0, cell_y0, cell_x1, cell_y1))
                    # Quick OCR with pytesseract if available
                    import pytesseract
                    import shutil
                    if shutil.which("tesseract") is None:
                        import os
                        tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                        if os.path.isfile(tess_path):
                            pytesseract.pytesseract.tesseract_cmd = tess_path
                    cell_text = pytesseract.image_to_string(cell_crop, config="--psm 6").strip()
                except Exception:
                    pass
            
            # Clean up the text
            cell_text = " ".join(cell_text.split())
            grid[ri][ci] = cell_text
    
    # Use first row as header if it overlaps with detected headers
    headers = structure.get("headers", [])
    if headers and n_rows > 1:
        df = pd.DataFrame(grid[1:], columns=grid[0])
    else:
        df = pd.DataFrame(grid)
    
    return df


def extract_tables(image: Image.Image, pdf_page=None, 
                   detection_conf: float = 0.7, 
                   structure_conf: float = 0.5) -> list[dict]:
    """
    Full TATR table extraction pipeline:
    1. Detect tables in the page image
    2. For each table, recognize structure (rows/cols)
    3. Extract cell content and build DataFrame
    
    Args:
        image: Full page PIL Image
        pdf_page: Optional fitz page object for native text extraction
        detection_conf: Confidence threshold for table detection
        structure_conf: Confidence threshold for structure recognition
    
    Returns:
        List of dicts with table_index, data, dataframe, markdown, engine
    """
    try:
        # Step 1: Detect tables
        table_bboxes = _detect_tables_in_image(image, confidence=detection_conf)
        
        if not table_bboxes:
            return []
        
        results = []
        for idx, table in enumerate(table_bboxes):
            bbox = table["bbox"]
            
            # Pad bbox slightly for better cropping
            pad = 5
            x0 = max(0, bbox[0] - pad)
            y0 = max(0, bbox[1] - pad)
            x1 = min(image.width, bbox[2] + pad)
            y1 = min(image.height, bbox[3] + pad)
            
            table_img = image.crop((x0, y0, x1, y1))
            
            # Step 2: Recognize structure
            structure = _recognize_structure(table_img, confidence=structure_conf)
            
            # Step 3: Build grid
            # Convert bbox to PDF coords if pdf_page is available
            pdf_bbox = None
            if pdf_page is not None:
                scale = pdf_page.rect.width / image.width
                pdf_bbox = [x0 * scale, y0 * scale, x1 * scale, y1 * scale]
            
            df = _build_grid(structure, table_img, pdf_page, pdf_bbox)
            
            if df.empty:
                # Fallback: return structure info
                df = pd.DataFrame([{
                    "rows_detected": len(structure["rows"]),
                    "columns_detected": len(structure["columns"]),
                    "detection_confidence": f"{table['confidence']:.3f}"
                }])
            
            markdown = df.to_markdown(index=False) if not df.empty else ""
            
            results.append({
                "table_index": idx + 1,
                "bbox": bbox,
                "detection_confidence": table["confidence"],
                "structure": {
                    "n_rows": len(structure["rows"]),
                    "n_cols": len(structure["columns"]),
                    "n_headers": len(structure["headers"])
                },
                "data": df.values.tolist(),
                "dataframe": df,
                "markdown": markdown,
                "engine": "TATR"
            })
        
        return results
        
    except ImportError as e:
        print(f"[ERROR] TATR dependencies missing: {e}")
        print("  Install with: pip install transformers torch torchvision")
        return []
    except Exception as e:
        print(f"[ERROR] TATR table extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return []
