"""
EasyOCR Wrapper
----------------
Uses EasyOCR deep learning models to extract text from document images.
"""

import numpy as np
from PIL import Image

_reader = None
_loaded_langs = None

def _load_reader(langs: list[str]):
    """Lazy initialize the EasyOCR Reader."""
    global _reader, _loaded_langs
    # Only re-initialize if the languages list changes
    if _reader is None or _loaded_langs != sorted(langs):
        try:
            import easyocr
            print(f"[OCR] Initializing EasyOCR Reader with languages: {langs}...")
            _reader = easyocr.Reader(langs)
            _loaded_langs = sorted(langs)
        except ImportError:
            print("[ERROR] easyocr is not installed. Please run: pip install easyocr")
            return None
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
    if langs is None:
        langs = ["en"]
        
    reader = _load_reader(langs)
    if reader is None:
        return {"full_text": "", "blocks": [], "engine": "EasyOCR", "error": "ImportError"}

    try:
        # Convert image to numpy array (EasyOCR requirement)
        img_np = np.array(image)
        results = reader.readtext(img_np)
        
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
