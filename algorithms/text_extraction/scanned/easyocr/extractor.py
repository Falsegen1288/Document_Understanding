"""
EasyOCR Wrapper
----------------
Uses EasyOCR deep learning models to extract text from document images.
"""

import numpy as np
from PIL import Image

_reader = None
_loaded_langs = None
RECOGNITION_CALL_COUNT = 0


def get_recognition_call_count() -> int:
    return RECOGNITION_CALL_COUNT


def _load_reader(langs: list[str]):
    """Lazy initialize and cache the EasyOCR Reader."""
    global _reader, _loaded_langs
    if _reader is None:
        try:
            import torch
            import easyocr
            gpu_avail = torch.cuda.is_available() and torch.cuda.device_count() > 0
            print(f"[EASYOCR INSTANTIATION] Instantiating SINGLE GLOBAL EasyOCR Reader instance (gpu={gpu_avail}) for languages: {langs}...", flush=True)
            try:
                _reader = easyocr.Reader(langs, gpu=gpu_avail)
            except Exception as e_gpu:
                print(f"[EASYOCR GPU FALLBACK] GPU init failed ({e_gpu}). Falling back to CPU mode (gpu=False)...", flush=True)
                _reader = easyocr.Reader(langs, gpu=False)

            _loaded_langs = sorted(langs)
        except ImportError:
            print("[ERROR] easyocr is not installed. Please run: pip install easyocr", flush=True)
            return None
    else:

        print(f"[EASYOCR REUSE] Reusing cached global EasyOCR Reader instance (Call #{RECOGNITION_CALL_COUNT + 1})!", flush=True)
    return _reader



def extract_text(image: Image.Image, langs: list[str] | None = None) -> dict:
    """
    Extract text from a page image using EasyOCR.
    
    Args:
        image: PIL Image of the page.
        langs: List of language codes. Defaults to ["en"].
        
    Returns:
        Dict containing full text and block coordinates.
    """
    global RECOGNITION_CALL_COUNT
    if langs is None:
        langs = ["en"]
        
    reader = _load_reader(langs)
    if reader is None:
        return {"full_text": "", "blocks": [], "engine": "EasyOCR", "error": "ImportError"}

    try:
        # Convert image to numpy array (EasyOCR requirement)
        img_np = np.array(image)
        RECOGNITION_CALL_COUNT += 1
        page_id = RECOGNITION_CALL_COUNT
        print(f"[READTEXT_CALL START] page={page_id} img_shape={img_np.shape}", flush=True)
        results = reader.readtext(img_np)
        print(f"[READTEXT_CALL END] page={page_id} num_detections={len(results)}", flush=True)

        
        blocks = []
        full_text_parts = []
        
        for r in results:
            # r format: (bbox points, text string, confidence score)
            # bbox points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            points, text, conf = r
            text = text.strip()
            
            if text:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                
                blocks.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": float(conf)
                })
                full_text_parts.append(text)
                
        return {
            "full_text": " ".join(full_text_parts),
            "blocks": blocks,
            "engine": "EasyOCR"
        }
    except Exception as e:
        print(f"[ERROR] EasyOCR failed: {e}")
        return {"full_text": "", "blocks": [], "engine": "EasyOCR", "error": str(e)}
