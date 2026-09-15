"""
Gemini Vision API Extractor
----------------------------
Sends diagram/figure crops and page images to Google Gemini Vision models
to extract detailed descriptions, data points, charts, tables, and text labels.
Uses persistent disk caching to avoid redundant API calls.
"""

import io
import os
import json
import hashlib
from typing import Optional, Dict, Any
from PIL import Image
from algorithms.config import GEMINI_API_KEY, GOOGLE_API_KEY

_FIGURE_CACHE: dict = {}
_FIGURE_CACHE_FILE = os.path.join(".cache", "figure_descriptions_cache.json")

# ── Generalized Master Prompt (Irrespective of Dataset Nature) ────────────────
GENERALIZED_MASTER_FIGURE_PROMPT = (
    "You are an expert multimodal document intelligence system. "
    "Perform an exhaustive, factual transcription and visual analysis of this image/figure crop from a document, "
    "regardless of whether the document is financial, scientific, medical, legal, commercial, or technical.\n\n"
    "Follow these structured instructions:\n"
    "1. VISUAL CLASSIFICATION:\n"
    "   - Identify the exact visual type (e.g., Bar Chart, Line Graph, Scatter Plot, Pie/Donut Chart, "
    "Flowchart, System Architecture Diagram, Workflow/Process Map, Schematic, Heatmap, Medical/Scientific Illustration, "
    "or Organizational Chart).\n\n"
    "2. TITLE, LABELS & UNITS:\n"
    "   - Transcribe all titles, subtitles, headers, and section captions verbatim.\n"
    "   - State all axis names (X and Y), scale intervals, and exact units of measurement "
    "(e.g., USD millions, EUR, %, basis points, mg/mL, dates, years, counts).\n"
    "   - List all legend entries, series names, colors, and pattern keys.\n\n"
    "3. VERBATIM DATA EXTRACTION:\n"
    "   - Extract ALL visible numerical values, coordinates, percentages, and data points verbatim.\n"
    "   - If comparative data or multiple series are present, structure them into a clean Markdown table "
    "mapping Category/Date -> Series -> Value.\n"
    "   - If a flowchart, architecture diagram, or process map, list each node/box and describe all directional arrows "
    "and logic flows sequentially from source to destination.\n\n"
    "4. KEY FINDINGS, TRENDS & ANNOTATIONS:\n"
    "   - Summarize the primary takeaways, trends (increases, declines, inflections), correlations, or anomalies shown.\n"
    "   - Transcribe all callout text boxes, footnotes, asterisked disclosures, and source notes verbatim.\n\n"
    "RULES: Do not extrapolate or guess values that are not visibly rendered. Maintain strict numerical precision."
)


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
    prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-3.6-flash",
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Sends figure image to Gemini Vision API, persisting results to disk cache.
    Uses GENERALIZED_MASTER_FIGURE_PROMPT by default, with optional context grounding.
    """
    effective_prompt = prompt if prompt is not None else GENERALIZED_MASTER_FIGURE_PROMPT
    if context and context.strip():
        effective_prompt = f"{effective_prompt}\n\n[SURROUNDING DOCUMENT CONTEXT]\n{context.strip()}"

    # 1. Check Disk Cache
    _load_figure_cache()
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    raw_b = img_bytes.getvalue()
    cache_key = hashlib.md5(f"{effective_prompt}:::".encode("utf-8") + raw_b).hexdigest()
    if cache_key in _FIGURE_CACHE:
        return _FIGURE_CACHE[cache_key]

    # 2. Resolve API Key from argument, algorithms.config, or environment (.env)
    key = api_key or GEMINI_API_KEY or GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if key:
        try:
            from google import genai
            g_client = genai.Client(api_key=key)
            models_to_try = [model, "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]
            # Deduplicate while preserving order
            seen = set()
            ordered_models = []
            for m in models_to_try:
                if m not in seen:
                    seen.add(m)
                    ordered_models.append(m)

            for m in ordered_models:
                try:
                    res = g_client.models.generate_content(
                        model=m,
                        contents=[image, effective_prompt]
                    )
                    if res and res.text:
                        desc = res.text.strip()
                        result = {
                            "description": desc,
                            "engine": "GeminiVision",
                            "model": m,
                            "prompt_used": effective_prompt
                        }
                        _FIGURE_CACHE[cache_key] = result
                        _save_figure_cache()
                        return result
                except Exception as e_m:
                    continue
        except Exception as e_gem:
            print(f"[WARN] Gemini Vision client initialization failed: {e_gem}", flush=True)

    return {
        "description": "Vision analysis unavailable.",
        "engine": "None",
        "model": "None",
        "error": "NoAvailableGeminiVisionBackend"
    }
