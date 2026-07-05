"""
Tesseract OCR Wrapper
----------------------
Uses pytesseract to extract text from document page images.
"""

from PIL import Image

def extract_text(image: Image.Image, lang: str = "eng", config: str = "") -> dict:
    """
    Extract text from a page image using Tesseract OCR.
    
    Args:
        image: PIL Image of the page.
        lang: Language model to use.
        config: Optional pytesseract configuration options.
        
    Returns:
        Dict with text elements:
            - full_text: full page text
            - blocks: list of word/line level coordinates
            - engine: 'Tesseract'
    """
    try:
        import pytesseract
        from pytesseract import Output
        
        # Auto-detect Tesseract binary on Windows if not in PATH
        import shutil, os, sys
        if shutil.which("tesseract") is None:
            # Common Windows install paths
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
            ]
            for path in candidates:
                if os.path.isfile(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Extract word details
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)
        full_text = pytesseract.image_to_string(image, lang=lang, config=config)
        
        blocks = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            conf = float(data['conf'][i])
            text = data['text'][i].strip()
            # Only keep high confidence non-empty blocks
            if conf > 0 and text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                blocks.append({
                    "text": text,
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": conf / 100.0
                })
                
        return {
            "full_text": full_text.strip(),
            "blocks": blocks,
            "engine": "Tesseract"
        }
    except ImportError:
        print("[ERROR] pytesseract is not installed. Please run: pip install pytesseract")
        return {"full_text": "", "blocks": [], "engine": "Tesseract", "error": "ImportError"}
    except Exception as e:
        print(f"[ERROR] Tesseract OCR failed: {e}")
        return {"full_text": "", "blocks": [], "engine": "Tesseract", "error": str(e)}
