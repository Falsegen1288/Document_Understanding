"""
Groq Vision API Extractor
--------------------------
Sends diagram/figure crops to Groq vision models to generate captions.
"""

import io
import base64
from PIL import Image
from algorithms.config import GROQ_API_KEY, GEMINI_API_KEY

def _encode_image(image: Image.Image) -> str:
    """Encode PIL image as base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

import os
import json
import hashlib

_FIGURE_CACHE: dict = {}
_FIGURE_CACHE_FILE = os.path.join(".cache", "figure_descriptions_cache.json")

def _load_figure_cache():
    global _FIGURE_CACHE
    if not _FIGURE_CACHE and os.path.exists(_FIGURE_CACHE_FILE):
        try:
            with open(_FIGURE_CACHE_FILE, "r", encoding="utf-8") as f:
                _FIGURE_CACHE = json.load(f)
        except Exception:
            _FIGURE_CACHE = {}

def _save_figure_cache():
    try:
        os.makedirs(os.path.dirname(_FIGURE_CACHE_FILE), exist_ok=True)
        with open(_FIGURE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_FIGURE_CACHE, f, indent=2)
    except Exception:
        pass

def describe_figure(
    image: Image.Image,
    prompt: str | None = None,
    api_key: str | None = None,
    model: str = "qwen/qwen3.8-27b"
) -> dict:
    """
    Sends figure image to Vision model (Gemini Vision with Groq fallback),
    persisting results to disk cache for zero redundant API calls.
    """
    if prompt is None:
        prompt = "Analyze this figure from a document. Describe what it shows, any data presented, labels, axes, and key takeaways."

    # 1. Check Disk Cache
    _load_figure_cache()
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    raw_b = img_bytes.getvalue()
    cache_key = hashlib.md5(f"{prompt}:::".encode("utf-8") + raw_b).hexdigest()
    if cache_key in _FIGURE_CACHE:
        return _FIGURE_CACHE[cache_key]

    # 2. Priority 1: Gemini Vision (fast, free tier, multimodal, reliable)
    gemini_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
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
                            "engine": "GeminiVision",
                            "model": m,
                            "prompt_used": prompt
                        }
                        _FIGURE_CACHE[cache_key] = result
                        _save_figure_cache()
                        return result
                except Exception:
                    continue
        except Exception as e_gem:
            print(f"[WARN] Gemini Vision attempt failed: {e_gem}")

    # 3. Priority 2: Groq Vision
    key = api_key if api_key else GROQ_API_KEY
    if key:
        if model not in ["qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]:
            model = "qwen/qwen3.8-27b"
        try:
            from groq import Groq
            client = Groq(api_key=key)
            base64_image = base64.b64encode(raw_b).decode("utf-8")
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                model=model,
                max_tokens=256
            )
            desc = chat_completion.choices[0].message.content.strip()
            result = {
                "description": desc,
                "engine": "Groq",
                "model": model,
                "prompt_used": prompt
            }
            _FIGURE_CACHE[cache_key] = result
            _save_figure_cache()
            return result
        except Exception as e:
            print(f"[ERROR] Groq Vision analysis failed: {e}")

    return {
        "description": "Vision analysis unavailable.",
        "engine": "None",
        "model": "None",
        "error": "NoAvailableVisionBackend"
    }


