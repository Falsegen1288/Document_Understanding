import os
from typing import Optional
from PIL import Image
import torch

os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/huggingface_cache")


import time

import io
import json
import hashlib
import time

_TEXT_OCR_CACHE: dict = {}
_TEXT_OCR_CACHE_FILE = os.path.join(".cache", "glm_ocr_text_cache.json")

def _load_text_ocr_cache():
    global _TEXT_OCR_CACHE
    if not _TEXT_OCR_CACHE and os.path.exists(_TEXT_OCR_CACHE_FILE):
        try:
            with open(_TEXT_OCR_CACHE_FILE, "r", encoding="utf-8") as f:
                _TEXT_OCR_CACHE = json.load(f)
        except Exception:
            _TEXT_OCR_CACHE = {}

def _save_text_ocr_cache():
    try:
        os.makedirs(os.path.dirname(_TEXT_OCR_CACHE_FILE), exist_ok=True)
        with open(_TEXT_OCR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_TEXT_OCR_CACHE, f, indent=2)
    except Exception:
        pass

class GLMOCRExtractor:
    """
    Vision-based OCR Extractor for GLM-OCR pipeline.
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
            print(f"[GLM-OCR] Enabled high-speed Gemini Vision engine (zero-VRAM, no torchvision bottleneck).", flush=True)
            self.is_degraded = False
            return

        t0 = time.perf_counter()
        print(f"[GLM-OCR SEGMENT A] [{time.strftime('%H:%M:%S')}] Starting model load attempt for '{self.MODEL_ID}' on {self.device}...", flush=True)

        try:
            from transformers import AutoProcessor
            from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model = GlmOcrForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()
            t_load = time.perf_counter() - t0
            print(f"[GLM-OCR SEGMENT B-SUCCESS] [{time.strftime('%H:%M:%S')}] Model loaded successfully in {t_load:.3f}s!", flush=True)
        except Exception as e:
            t_fail = time.perf_counter() - t0
            print(f"[GLM-OCR SEGMENT B-FAIL] [{time.strftime('%H:%M:%S')}] Failed after {t_fail:.3f}s ({e}). Degrading to baseline path...", flush=True)
            self.is_degraded = True


    def extract_text(self, image: Image.Image) -> str:
        # 1. Check Disk Cache
        _load_text_ocr_cache()
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        cache_key = hashlib.md5(b"glm_ocr_text:::" + img_buf.getvalue()).hexdigest()
        if cache_key in _TEXT_OCR_CACHE:
            return _TEXT_OCR_CACHE[cache_key]

        # 2. Gemini Vision Pass
        if self.use_gemini:
            try:
                from google import genai
                g_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                prompt = (
                    "Extract all text, numbers, headings, tables, and visual elements from this document page verbatim. "
                    "Transcribe all charts, line graphs, bar charts, diagrams, logos, and graphic legends in full detail. "
                    "Preserve structural headings, tabular format, and data relationships."
                )
                for m in ["gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash"]:
                    try:
                        res = g_client.models.generate_content(
                            model=m,
                            contents=[image, prompt]
                        )
                        if res and res.text:
                            text_out = res.text.strip()
                            _TEXT_OCR_CACHE[cache_key] = text_out
                            _save_text_ocr_cache()
                            return text_out
                    except Exception:
                        continue
            except Exception as e_gem:
                print(f"[GLM-OCR Gemini WARNING] Vision call failed ({e_gem})", flush=True)

        if self.is_degraded or self.model is None or self.processor is None:
            return ""

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Extract all text from this document page."}
                    ]
                }
            ]
            prompt_str = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(images=image, text=prompt_str, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=2048)
            new_tokens = out[:, input_len:]
            res_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            del inputs, out, new_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _TEXT_OCR_CACHE[cache_key] = res_text
            _save_text_ocr_cache()
            return res_text
        except Exception as e:
            print(f"[GLM-OCR WARNING] Inference failed: {e}", flush=True)
            return ""
