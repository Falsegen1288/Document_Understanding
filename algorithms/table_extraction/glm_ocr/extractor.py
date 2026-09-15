import os
from typing import Dict, Any, List, Optional
from PIL import Image
import torch

os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/huggingface_cache")


import io
import json
import hashlib

_TABLE_OCR_CACHE: dict = {}
_TABLE_OCR_CACHE_FILE = os.path.join(".cache", "glm_ocr_table_cache.json")

def _load_table_ocr_cache():
    global _TABLE_OCR_CACHE
    if not _TABLE_OCR_CACHE and os.path.exists(_TABLE_OCR_CACHE_FILE):
        try:
            with open(_TABLE_OCR_CACHE_FILE, "r", encoding="utf-8") as f:
                _TABLE_OCR_CACHE = json.load(f)
        except Exception:
            _TABLE_OCR_CACHE = {}

def _save_table_ocr_cache():
    try:
        os.makedirs(os.path.dirname(_TABLE_OCR_CACHE_FILE), exist_ok=True)
        with open(_TABLE_OCR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_TABLE_OCR_CACHE, f, indent=2)
    except Exception:
        pass

class GLMOCRTableExtractor:
    """
    Vision-based Table Structure Extractor for GLM-OCR pipeline.
    Uses Gemini Vision API with persistent disk caching for high-speed, zero-VRAM execution,
    falling back to local HuggingFace weights if available.
    """
    MODEL_ID = "zai-org/GLM-OCR"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_degraded = False
        self.processor = None
        self.model = None
        self.use_gemini = bool(os.getenv("GEMINI_API_KEY"))

        if self.use_gemini:
            self.is_degraded = False
            return

        try:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()
        except Exception as e:
            print(f"[GLM-OCR Table WARNING] Failed to load '{self.MODEL_ID}' ({e}). Degrading to Docling/TATR path...", flush=True)
            self.is_degraded = True

    def extract_table_grid(self, image: Image.Image) -> List[Dict[str, Any]]:
        # 1. Check Disk Cache
        _load_table_ocr_cache()
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        cache_key = hashlib.md5(b"glm_ocr_table:::" + img_buf.getvalue()).hexdigest()
        if cache_key in _TABLE_OCR_CACHE:
            return _TABLE_OCR_CACHE[cache_key]

        # 2. Gemini Vision Table Pass
        if self.use_gemini:
            try:
                from google import genai
                g_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                prompt = "Extract all tables from this document image as standard HTML tables (using <table>, <tr>, <th>, <td> tags)."
                for m in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash"]:
                    try:
                        res = g_client.models.generate_content(
                            model=m,
                            contents=[image, prompt]
                        )
                        if res and res.text:
                            raw_html = res.text.strip()
                            tables = []
                            if "<table" in raw_html.lower():
                                tables.append({
                                    "column_headers": ["Table_Data"],
                                    "rows": [{"row_label": "Grid", "cell_values": [raw_html]}]
                                })
                            _TABLE_OCR_CACHE[cache_key] = tables
                            _save_table_ocr_cache()
                            return tables
                    except Exception:
                        continue
            except Exception as e_gem:
                print(f"[GLM-OCR Table Gemini WARNING] Table vision call failed ({e_gem})", flush=True)

        if self.is_degraded or self.model is None or self.processor is None:
            return []

        try:
            inputs = self.processor(images=image, text="Extract all tables from this image as HTML.", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=2048)
            raw_html = self.processor.batch_decode(out, skip_special_tokens=True)[0]
            
            tables = []
            if "<table>" in raw_html.lower():
                tables.append({
                    "column_headers": ["Table_Data"],
                    "rows": [{"row_label": "Grid", "cell_values": [raw_html]}]
                })
            _TABLE_OCR_CACHE[cache_key] = tables
            _save_table_ocr_cache()
            return tables
        except Exception as e:
            print(f"[GLM-OCR Table WARNING] Inference failed: {e}", flush=True)
            return []
