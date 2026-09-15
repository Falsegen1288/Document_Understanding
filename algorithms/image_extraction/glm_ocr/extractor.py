"""
GLM-OCR Vision / Figure Extractor
----------------------------------
Multimodal visual analysis for charts, diagrams, figures, and logos in the GLM-OCR pipeline.
Uses Gemini Vision API with persistent disk caching for zero-VRAM, high-speed execution,
falling back to local HuggingFace weights if available.
"""

import os
import io
import json
import hashlib
from typing import Optional, Dict, Any
from PIL import Image
import torch

_FIGURE_OCR_CACHE: dict = {}
_FIGURE_OCR_CACHE_FILE = os.path.join(".cache", "glm_ocr_figure_cache.json")
_SHARED_FIGURE_CACHE_FILE = os.path.join(".cache", "figure_descriptions_cache.json")


def _load_figure_cache():
    global _FIGURE_OCR_CACHE
    if not _FIGURE_OCR_CACHE:
        # Check dedicated cache first
        if os.path.exists(_FIGURE_OCR_CACHE_FILE):
            try:
                with open(_FIGURE_OCR_CACHE_FILE, "r", encoding="utf-8") as f:
                    _FIGURE_OCR_CACHE = json.load(f)
            except Exception:
                _FIGURE_OCR_CACHE = {}
        # Also load from shared cache if present
        if os.path.exists(_SHARED_FIGURE_CACHE_FILE):
            try:
                with open(_SHARED_FIGURE_CACHE_FILE, "r", encoding="utf-8") as f:
                    shared = json.load(f)
                    for k, v in shared.items():
                        if k not in _FIGURE_OCR_CACHE:
                            _FIGURE_OCR_CACHE[k] = v
            except Exception:
                pass


def _save_figure_cache():
    try:
        os.makedirs(os.path.dirname(_FIGURE_OCR_CACHE_FILE), exist_ok=True)
        with open(_FIGURE_OCR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_FIGURE_OCR_CACHE, f, indent=2)
    except Exception:
        pass


class GLMOCRFigureExtractor:
    """
    Vision-based Figure and Visual Element Extractor for GLM-OCR pipeline.
    Uses Gemini Vision API with persistent disk caching for high-speed execution.
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
            from transformers import AutoProcessor
            from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self.model = GlmOcrForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device).eval()
        except Exception as e:
            print(f"[GLM-OCR Figure WARNING] Failed to load local '{self.MODEL_ID}' ({e}). Running in fallback mode.", flush=True)
            self.is_degraded = True

    def describe_figure(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts full visual detail, labels, data, and structure from a figure image.
        """
        if prompt is None:
            prompt = (
                "Transcribe and extract ALL charts, figures, diagrams, tables, rows, numbers, "
                "percentages, taglines, and data values in this image in full detail. "
                "List all labels, axes, and entries verbatim."
            )

        # 1. Check Disk Cache
        _load_figure_cache()
        img_buf = io.BytesIO()
        image.save(img_buf, format="PNG")
        raw_b = img_buf.getvalue()
        cache_key = hashlib.md5(f"{prompt}:::".encode("utf-8") + raw_b).hexdigest()
        if cache_key in _FIGURE_OCR_CACHE:
            return _FIGURE_OCR_CACHE[cache_key]

        # 2. Priority 1: Gemini Vision
        gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                g_client = genai.Client(api_key=gemini_key)
                for m in ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]:
                    try:
                        res = g_client.models.generate_content(
                            model=m,
                            contents=[image, prompt]
                        )
                        if res and res.text:
                            desc = res.text.strip()
                            result = {
                                "description": desc,
                                "engine": "GLM_GeminiVision",
                                "model": m,
                                "prompt_used": prompt
                            }
                            _FIGURE_OCR_CACHE[cache_key] = result
                            _save_figure_cache()
                            return result
                    except Exception:
                        continue
            except Exception as e_gem:
                print(f"[GLM-OCR Figure Gemini WARNING] Vision call failed ({e_gem})", flush=True)

        # 3. Priority 2: Local GLM-OCR model if available
        if not self.is_degraded and self.model is not None and self.processor is not None:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                prompt_str = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(images=image, text=prompt_str, return_tensors="pt").to(self.device)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=1024)
                new_tokens = out[:, input_len:]
                res_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                del inputs, out, new_tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = {
                    "description": res_text,
                    "engine": "GLM-OCR-Local",
                    "model": self.MODEL_ID,
                    "prompt_used": prompt
                }
                _FIGURE_OCR_CACHE[cache_key] = result
                _save_figure_cache()
                return result
            except Exception as e:
                print(f"[GLM-OCR Figure WARNING] Local inference failed: {e}", flush=True)

        return {
            "description": "Vision analysis unavailable.",
            "engine": "None",
            "model": "None",
            "error": "NoAvailableVisionBackend"
        }


_DEFAULT_EXTRACTOR: Optional[GLMOCRFigureExtractor] = None


def describe_figure(
    image: Image.Image,
    prompt: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Module-level helper for GLM-OCR figure description."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = GLMOCRFigureExtractor()
    return _DEFAULT_EXTRACTOR.describe_figure(image, prompt=prompt, api_key=api_key)
