import os
import json
import re
import math
from typing import List, Dict, Any, Optional
from algorithms.config import GEMINI_API_KEY, GOOGLE_API_KEY
from ragas.llms import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding


class GeminiRagasLLM(InstructorBaseRagasLLM):
    """
    Ragas-compatible LLM wrapper powered by Google Gemini via google.genai Client.
    """
    def __init__(self, model: str = "gemini-3.5-flash-lite", api_key: Optional[str] = None):
        self.model = model or "gemini-3.5-flash-lite"
        self.api_key = api_key or GEMINI_API_KEY or GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[GeminiRagasLLM WARNING] Could not initialize Gemini client: {e}", flush=True)

    def generate(self, prompt: str, response_model: Any) -> Any:
        if self.client is None:
            raise RuntimeError("Gemini client not initialized for Ragas")

        from google.genai import types
        schema = response_model.model_json_schema()
        full_prompt = (
            f"{prompt}\n\n"
            f"You MUST respond with valid JSON strictly adhering to this JSON schema:\n"
            f"{json.dumps(schema)}"
        )

        models_to_try = [self.model, "gemini-3.5-flash-lite", "gemini-3.6-flash"]
        seen = set()
        for m in models_to_try:
            if m in seen:
                continue
            seen.add(m)
            for attempt in range(3):
                try:
                    res = self.client.models.generate_content(
                        model=m,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json"
                        )
                    )
                    if res and res.text:
                        txt = res.text.strip()
                        if txt.startswith("```"):
                            txt = re.sub(r"^```json\s*", "", txt)
                            txt = re.sub(r"\s*```$", "", txt)
                        return response_model.model_validate_json(txt)
                except Exception as e_gen:
                    err_s = str(e_gen)
                    if ("429" in err_s or "RESOURCE_EXHAUSTED" in err_s) and attempt < 2:
                        import time
                        time.sleep(5.0 * (attempt + 1))
                        continue
                    break

        raise RuntimeError("Failed all Gemini models for Ragas structured output generation")

    async def agenerate(self, prompt: str, response_model: Any) -> Any:
        return self.generate(prompt, response_model)


class LocalRagasEmbedding(BaseRagasEmbedding):
    """
    Ragas-compatible embedding wrapper backed by local SentenceTransformer/EmbeddingStage.
    """
    def __init__(self):
        super().__init__()
        from benchmark_harness.stages.embedding import EmbeddingStage
        self.stage = EmbeddingStage()

    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        vec = self.stage.encode([text])[0]
        return vec.tolist()

    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        return self.embed_text(text, **kwargs)

    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.stage.encode(texts)
        return vecs.tolist()

    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return self.embed_texts(texts, **kwargs)


class RagasEvaluator:
    """
    Ragas Evaluator: Runs Faithfulness, Answer Relevancy, Context Precision, and Context Recall
    using Google Gemini and local semantic embeddings.
    """
    def __init__(self, model_name: str = "gemini-3.5-flash-lite", api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY or GOOGLE_API_KEY
        self.model_name = model_name or "gemini-3.5-flash-lite"
        self.llm = None
        self.embeddings = None

        try:
            self.llm = GeminiRagasLLM(model=self.model_name, api_key=self.api_key)
            self.embeddings = LocalRagasEmbedding()
            print(f"[Ragas] Initialized RagasEvaluator with model '{self.model_name}'.", flush=True)
        except Exception as e:
            print(f"[Ragas WARNING] Failed to initialize RagasEvaluator: {e}", flush=True)

    def evaluate_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a list of query records:
        Each record must have:
        - "input" / "query": str
        - "actual_output" / "prediction": str
        - "expected_output" / "ground_truth": str
        - "retrieval_context": List[str]
        """
        if not self.llm or not records:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "individual_scores": [],
                "error": "Ragas evaluator unavailable or empty records"
            }

        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

        faith_m = Faithfulness(llm=self.llm)
        rel_m = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        prec_m = ContextPrecision(llm=self.llm)
        recall_m = ContextRecall(llm=self.llm)

        import time
        results = []

        for idx, item in enumerate(records):
            q = str(item.get("input") or item.get("query", "")).strip()
            ans = str(item.get("actual_output") or item.get("prediction", "")).strip() or "NOT_FOUND"
            gt = str(item.get("expected_output") or item.get("ground_truth", "")).strip() or "N/A"
            contexts = item.get("retrieval_context") or ["N/A"]
            if isinstance(contexts, str):
                contexts = [contexts]

            # Pacing delay between queries to respect RPM rate limits
            if idx > 0:
                time.sleep(2.0)

            # 1. Faithfulness
            f_val = None
            try:
                res = faith_m.score(user_input=q, response=ans, retrieved_contexts=contexts)
                if res is not None and res.value is not None and not math.isnan(float(res.value)):
                    f_val = float(res.value)
            except Exception as e_f:
                pass
            time.sleep(1.0)

            # 2. Answer Relevancy
            r_val = None
            try:
                res = rel_m.score(user_input=q, response=ans)
                if res is not None and res.value is not None and not math.isnan(float(res.value)):
                    r_val = float(res.value)
            except Exception as e_r:
                pass
            time.sleep(1.0)

            # 3. Context Precision
            p_val = None
            try:
                res = prec_m.score(user_input=q, reference=gt, retrieved_contexts=contexts)
                if res is not None and res.value is not None and not math.isnan(float(res.value)):
                    p_val = float(res.value)
            except Exception as e_p:
                pass
            time.sleep(1.0)

            # 4. Context Recall
            rec_val = None
            try:
                res = recall_m.score(user_input=q, retrieved_contexts=contexts, reference=gt)
                if res is not None and res.value is not None and not math.isnan(float(res.value)):
                    rec_val = float(res.value)
            except Exception as e_rec:
                pass

            results.append({
                "query": q,
                "faithfulness": round(f_val, 4) if f_val is not None else None,
                "answer_relevancy": round(r_val, 4) if r_val is not None else None,
                "context_precision": round(p_val, 4) if p_val is not None else None,
                "context_recall": round(rec_val, 4) if rec_val is not None else None,
            })

        # Calculate averages strictly over legitimate, non-null scores (excluding 429/errors)
        valid_f = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
        valid_r = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]
        valid_p = [r["context_precision"] for r in results if r["context_precision"] is not None]
        valid_rec = [r["context_recall"] for r in results if r["context_recall"] is not None]

        return {
            "faithfulness": round(sum(valid_f) / len(valid_f), 4) if valid_f else None,
            "answer_relevancy": round(sum(valid_r) / len(valid_r), 4) if valid_r else None,
            "context_precision": round(sum(valid_p) / len(valid_p), 4) if valid_p else None,
            "context_recall": round(sum(valid_rec) / len(valid_rec), 4) if valid_rec else None,
            "valid_queries": len(valid_f),
            "total_queries": len(records),
            "error_queries": len(records) - len(valid_f),
            "individual_scores": results
        }
