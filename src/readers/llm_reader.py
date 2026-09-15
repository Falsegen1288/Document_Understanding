import os
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from algorithms.config import GROQ_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY

# Short-form: TAT-DQA-style extractive answers (single value/number/span)
PROMPT_TEMPLATE_SHORT = """You are an expert document understanding system.
Answer the question using ONLY the provided context. If the answer is not present, return "NOT_FOUND".

Context:
{context}

Question: {query}

Answer concisely (span, number, or short phrase only, no extra explanation):"""

# Multi-span: TAT-DQA multi-span answers (more than one discrete value expected)
PROMPT_TEMPLATE_MULTI_SPAN = """You are an expert document understanding system.
Answer the question using ONLY the provided context. If the answer is not present, return "NOT_FOUND".

Context:
{context}

Question: {query}

This question expects MULTIPLE distinct values as the answer (e.g. one per year, category, or item compared).
List each value, in the order the question asks for them, separated by commas. No explanation."""

# Prose: UniDoc-style synthesized/explanatory answers
PROMPT_TEMPLATE_PROSE = """You are an expert document understanding system.
Answer the question using ONLY the provided context. If the answer is not present, return "NOT_FOUND".

Context:
{context}

Question: {query}

Answer in 1-3 full sentences, synthesizing the relevant facts and numerical details from the context. Do not pad with information the question didn't ask for. No conversational preamble."""

# TAT-DQA answer_type values that call for a short/extractive answer.
_SHORT_FORM_TYPES = {"span", "count", "arithmetic"}
_MULTI_SPAN_TYPES = {"multi-span"}


def _select_prompt_template(query_type: Optional[str]) -> str:
    q = str(query_type or "").strip().lower()
    if q in _MULTI_SPAN_TYPES:
        return PROMPT_TEMPLATE_MULTI_SPAN
    if q in _SHORT_FORM_TYPES:
        return PROMPT_TEMPLATE_SHORT
    return PROMPT_TEMPLATE_PROSE


class LLMReader:
    """
    Generative LLM-based Reader using Google Gemini 3.6 Flash as the primary engine,
    with persistent disk caching and Groq fallback.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.getenv("READER_MODEL", "gemini-3.6-flash")
        self.gemini_key = api_key or GEMINI_API_KEY or GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.groq_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")

        self.gemini_client = None
        if self.gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.gemini_key)
            except Exception as e:
                print(f"[LLMReader WARNING] Could not initialize Gemini client: {e}", flush=True)

        self.groq_client = None
        if self.groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_key)
            except Exception as e:
                print(f"[LLMReader WARNING] Could not initialize Groq client: {e}", flush=True)

        self._cache: Dict[str, Any] = {}
        self._cache_file = os.path.join(".cache", "llm_reader_cache.json")
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
                json.dump(self._cache, f)
        except Exception:
            pass

    def answer(self, query: str, context_chunks: List[str], query_type: Optional[str] = None) -> Dict[str, Any]:
        if not context_chunks:
            return {"answer": "NOT_FOUND", "not_found": True, "raw_response": "NOT_FOUND"}

        context = "\n\n".join(context_chunks)
        if len(context) > 64000:
            context = context[:64000]

        template = _select_prompt_template(query_type)
        prompt = template.format(context=context, query=query)

        cache_key = hashlib.md5(f"{self.model}:{prompt}".encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 1. Primary Path: Google Gemini 3.6 Flash
        if self.gemini_client is not None:
            from google.genai import types
            models_to_try = [self.model, "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.5-flash-lite"]
            seen = set()
            for m in models_to_try:
                if m in seen:
                    continue
                seen.add(m)
                try:
                    res = self.gemini_client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512)
                    )
                    if res and res.text:
                        raw = res.text.strip()
                        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                        not_found = clean.upper().startswith("NOT_FOUND")
                        result = {
                            "answer": clean,
                            "not_found": not_found,
                            "raw_response": raw,
                            "model_used": m,
                            "provider": "gemini"
                        }
                        self._cache[cache_key] = result
                        if len(self._cache) % 10 == 0:
                            self._save_cache()
                        return result
                except Exception as e_gem:
                    continue

        # 2. Secondary Fallback Path: Groq API
        if self.groq_client is not None:
            try:
                groq_model = "qwen/qwen3.8-27b" if "gemini" in self.model else self.model
                resp = self.groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                raw = resp.choices[0].message.content.strip()
                clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                not_found = clean.upper().startswith("NOT_FOUND")
                result = {
                    "answer": clean,
                    "not_found": not_found,
                    "raw_response": raw,
                    "model_used": groq_model,
                    "provider": "groq"
                }
                self._cache[cache_key] = result
                self._save_cache()
                return result
            except Exception as e_groq:
                pass

        return {
            "answer": "",
            "not_found": True,
            "raw_response": "All LLM reader providers failed or unconfigured.",
            "error": "NoLLMReaderAvailable"
        }
