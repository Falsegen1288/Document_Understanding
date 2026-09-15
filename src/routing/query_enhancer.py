import os
import re
import json
import hashlib
from typing import Dict, Any, Optional
from algorithms.config import GEMINI_API_KEY, GOOGLE_API_KEY

QUERY_ENHANCER_PROMPT = """You are an expert retrieval-augmented document understanding query optimizer.
Analyze the user query below and return a JSON object with three fields:
1. "search_query": An optimized string of dense/sparse search keywords, key entities, financial or domain metrics, and fiscal periods specifically designed to retrieve relevant tables, paragraphs, and figures from a vector/BM25 database.
2. "structured_directive": A structured, step-by-step instruction for a Reader LLM specifying:
   - The primary objective and target entity/metric.
   - Exactly what figures/data points to extract from the context.
   - Any mathematical calculation, comparison, or synthesis required.
   - The required output formatting (concise span, comma-separated list, or 1-2 explanatory sentences).
3. "query_type": One of ["arithmetic", "span", "multi-span", "prose"].

User Query: "{query}"

Output ONLY a valid JSON object matching this schema:
{{
  "search_query": "...",
  "structured_directive": "...",
  "query_type": "..."
}}"""


class QueryEnhancer:
    """
    QueryEnhancer: Transforms raw user queries via Gemini 3.6 Flash into:
    1. Search keywords optimized for Qdrant Dense HNSW and BM25 Sparse retrieval.
    2. Structured reasoning directives that guide the Reader LLM to produce accurate answers.
    Includes persistent disk caching in .cache/query_enhancer_cache.json.
    """

    def __init__(self, model: str = "gemini-3.6-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or GEMINI_API_KEY or GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.gemini_client = None

        if self.api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[QueryEnhancer WARNING] Could not initialize Gemini client: {e}", flush=True)

        self._cache_file = os.path.join(".cache", "query_enhancer_cache.json")
        self._cache: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception:
            self._cache = {}

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def enhance(self, query: str) -> Dict[str, str]:
        """
        Enhances raw query. Returns a dict:
        {
            "original_query": query,
            "search_query": search_query,
            "structured_directive": structured_directive,
            "query_type": query_type
        }
        """
        cache_key = hashlib.md5(f"{self.model}:{query}".encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        fallback = {
            "original_query": query,
            "search_query": query,
            "structured_directive": query,
            "query_type": "prose"
        }

        if self.gemini_client is None:
            return fallback

        from google.genai import types
        prompt = QUERY_ENHANCER_PROMPT.format(query=query)

        models_to_try = [self.model, "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash"]
        seen = set()

        for m in models_to_try:
            if m in seen:
                continue
            seen.add(m)
            try:
                res = self.gemini_client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=512,
                        response_mime_type="application/json"
                    )
                )
                if res and res.text:
                    cleaned_text = res.text.strip()
                    if cleaned_text.startswith("```"):
                        cleaned_text = re.sub(r"^```json\s*", "", cleaned_text)
                        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
                    parsed = json.loads(cleaned_text)
                    result = {
                        "original_query": query,
                        "search_query": parsed.get("search_query", query).strip(),
                        "structured_directive": parsed.get("structured_directive", query).strip(),
                        "query_type": parsed.get("query_type", "prose").strip().lower(),
                        "model_used": m
                    }
                    self._cache[cache_key] = result
                    self._save_cache()
                    return result
            except Exception as e:
                continue

        return fallback
