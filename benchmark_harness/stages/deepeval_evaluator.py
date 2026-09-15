import os
import copy
from typing import List, Dict, Any, Optional
from algorithms.config import GEMINI_API_KEY, GOOGLE_API_KEY


class DeepEvalEvaluator:
    """
    DeepEval Evaluator: Runs Faithfulness, Answer Relevancy, and Contextual Precision
    using Google Gemini 3.6 Flash as the evaluation judge.
    """

    def __init__(self, model_name: str = "gemini-3.6-flash", api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY or GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.model = None

        try:
            from deepeval.models import GeminiModel
            from deepeval.models.llms.gemini_model import GEMINI_MODELS_DATA

            # Register gemini-3.6-flash and gemini-3.5-flash-lite in DeepEval model registry
            base_data = GEMINI_MODELS_DATA.get("gemini-2.5-flash") or list(GEMINI_MODELS_DATA.values())[0]
            if "gemini-3.6-flash" not in GEMINI_MODELS_DATA:
                GEMINI_MODELS_DATA["gemini-3.6-flash"] = copy.deepcopy(base_data)
            if "gemini-3.5-flash-lite" not in GEMINI_MODELS_DATA:
                GEMINI_MODELS_DATA["gemini-3.5-flash-lite"] = copy.deepcopy(base_data)

            # Check if primary model works or fallback to gemini-3.5-flash-lite
            chosen_model = self.model_name
            try:
                test_m = GeminiModel(model=chosen_model, api_key=self.api_key)
                test_m.generate("test")
            except Exception:
                chosen_model = "gemini-3.5-flash-lite"

            self.model = GeminiModel(model=chosen_model, api_key=self.api_key)
            print(f"[DeepEval] Initialized GeminiModel judge with model '{chosen_model}'.", flush=True)
        except Exception as e:
            print(f"[DeepEval WARNING] Could not initialize GeminiModel judge: {e}", flush=True)

    def evaluate_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates a list of query records:
        Each record must have:
        - "input" / "query": str
        - "actual_output" / "prediction": str
        - "expected_output" / "ground_truth": str
        - "retrieval_context": List[str]
        """
        if not self.model or not records:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "contextual_precision": 0.0,
                "individual_scores": [],
                "error": "DeepEval judge unavailable or empty records"
            }

        import time
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric

        faith_metric = FaithfulnessMetric(threshold=0.5, model=self.model)
        rel_metric = AnswerRelevancyMetric(threshold=0.5, model=self.model)
        prec_metric = ContextualPrecisionMetric(threshold=0.5, model=self.model)

        results = []

        for idx, r in enumerate(records):
            q = r.get("input") or r.get("query", "")
            actual = str(r.get("actual_output") or r.get("prediction", "")).strip() or "NOT_FOUND"
            expected = str(r.get("expected_output") or r.get("ground_truth", "")).strip() or "N/A"
            ctx = r.get("retrieval_context") or ["N/A"]
            if isinstance(ctx, str):
                ctx = [ctx]

            test_case = LLMTestCase(
                input=q,
                actual_output=actual,
                expected_output=expected,
                retrieval_context=ctx
            )

            # Pacing delay between queries to respect RPM rate limits
            if idx > 0:
                time.sleep(2.0)

            # Helper for robust measurement with backoff retry
            def _measure_with_retry(metric_obj, case_obj):
                for attempt in range(3):
                    try:
                        metric_obj.measure(case_obj)
                        score = float(metric_obj.score if metric_obj.score is not None else 0.0)
                        reason = str(metric_obj.reason or "")
                        return score, reason, None
                    except Exception as exc:
                        err_str = str(exc)
                        if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < 2:
                            time.sleep(5.0 * (attempt + 1))
                            continue
                        return None, f"Metric error: {err_str}", err_str
                return None, "Max retries exceeded", "RESOURCE_EXHAUSTED"

            # 1. Faithfulness
            f_score, f_reason, _ = _measure_with_retry(faith_metric, test_case)
            time.sleep(1.0)

            # 2. Answer Relevancy
            r_score, r_reason, _ = _measure_with_retry(rel_metric, test_case)
            time.sleep(1.0)

            # 3. Contextual Precision
            p_score, p_reason, _ = _measure_with_retry(prec_metric, test_case)

            results.append({
                "query": q,
                "faithfulness": round(f_score, 4) if f_score is not None else None,
                "faithfulness_reason": f_reason,
                "answer_relevancy": round(r_score, 4) if r_score is not None else None,
                "answer_relevancy_reason": r_reason,
                "contextual_precision": round(p_score, 4) if p_score is not None else None,
                "contextual_precision_reason": p_reason,
            })

        # Calculate averages strictly over legitimate, non-null scores (excluding 429/errors)
        valid_f = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
        valid_r = [r["answer_relevancy"] for r in results if r["answer_relevancy"] is not None]
        valid_p = [r["contextual_precision"] for r in results if r["contextual_precision"] is not None]

        return {
            "faithfulness": round(sum(valid_f) / len(valid_f), 4) if valid_f else None,
            "answer_relevancy": round(sum(valid_r) / len(valid_r), 4) if valid_r else None,
            "contextual_precision": round(sum(valid_p) / len(valid_p), 4) if valid_p else None,
            "valid_queries": len(valid_f),
            "total_queries": len(records),
            "error_queries": len(records) - len(valid_f),
            "individual_scores": results
        }
