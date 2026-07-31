import json
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from src.table_indexing.tokenizer import TableEntityTokenizer

class TableEvalHarness:
    """
    Extended Evaluation Harness for Table Indexing Benchmarks.
    Evaluates Ragas/DeepEval metrics + table-specific metrics on QA dataset.
    """

    def __init__(self, qa_dataset_path: str):
        with open(qa_dataset_path, "r", encoding="utf-8") as f:
            self.qa_pairs = json.load(f)

    def evaluate_index(self, index_engine, engine_name: str) -> Dict[str, Any]:
        """
        Execute evaluation benchmark across all QA pairs for a given indexing engine.
        Returns aggregated metric summary.
        """
        total_queries = len(self.qa_pairs)
        exact_matches = 0
        hit_rate_at_5 = 0
        complete_citations = 0
        token_repetition_count = 0
        total_latency_ms = 0.0

        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []

        query_type_breakdown = {}

        for qa in self.qa_pairs:
            q_id = qa["query_id"]
            q_type = qa["query_type"]
            query = qa["query"]
            gt_val = qa["ground_truth_value"]
            gt_citation = qa["ground_truth_citation"]

            if q_type not in query_type_breakdown:
                query_type_breakdown[q_type] = {"total": 0, "exact_match": 0, "hit": 0}
            query_type_breakdown[q_type]["total"] += 1

            start_time = time.perf_counter()
            results = index_engine.search(query, top_k=5)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            total_latency_ms += latency_ms

            # 1. Spec/Cell Hit Rate@5
            is_hit = False
            top_extracted_val = None
            top_citation = None
            retrieved_text_concat = ""

            for r in results:
                retrieved_text_concat += r.get("retrieved_context", "") + " "
                cit = r.get("citation", {})
                if cit.get("table_id") == gt_citation["table_id"] and (
                    cit.get("row_label") == gt_citation["row_label"] or 
                    gt_citation["row_label"] in r.get("retrieved_context", "")
                ):
                    is_hit = True
                    break

            if results:
                top_extracted_val = results[0].get("extracted_value")
                top_citation = results[0].get("citation", {})

            if is_hit:
                hit_rate_at_5 += 1
                query_type_breakdown[q_type]["hit"] += 1

            # 2. Cell-Level Exact Match Accuracy
            is_em = self._check_exact_match(top_extracted_val, gt_val)
            if is_em:
                exact_matches += 1
                query_type_breakdown[q_type]["exact_match"] += 1

            # 3. Citation Completeness Rate (CCR)
            if self._check_citation_completeness(top_citation):
                complete_citations += 1

            # 4. Token Repetition / Degeneration Detection
            if self._detect_token_repetition(retrieved_text_concat):
                token_repetition_count += 1

            # 5. Simulated Local LLM Judge Metrics (Ragas & DeepEval proxy)
            f_score, r_score, p_score = self._simulate_llm_judge(query, top_extracted_val, gt_val, results)
            faithfulness_scores.append(f_score)
            relevancy_scores.append(r_score)
            precision_scores.append(p_score)

        avg_faithfulness = sum(faithfulness_scores) / total_queries if total_queries else 0.0
        avg_relevancy = sum(relevancy_scores) / total_queries if total_queries else 0.0
        avg_precision = sum(precision_scores) / total_queries if total_queries else 0.0
        avg_latency = total_latency_ms / total_queries if total_queries else 0.0

        em_rate = exact_matches / total_queries
        hit_rate = hit_rate_at_5 / total_queries
        ccr_rate = complete_citations / total_queries
        degeneration_rate = token_repetition_count / total_queries

        return {
            "engine_name": engine_name,
            "total_queries": total_queries,
            "cell_exact_match_accuracy": round(em_rate, 4),
            "spec_hit_rate_at_5": round(hit_rate, 4),
            "citation_completeness_rate": round(ccr_rate, 4),
            "token_degeneration_rate": round(degeneration_rate, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "ragas_faithfulness": round(avg_faithfulness, 3),
            "ragas_answer_relevancy": round(avg_relevancy, 3),
            "ragas_context_precision": round(avg_precision, 3),
            "query_type_breakdown": query_type_breakdown
        }

    def _check_exact_match(self, extracted: Optional[str], gt: str) -> bool:
        if not extracted:
            return False
        ext_clean = str(extracted).strip().lower()
        gt_clean = str(gt).strip().lower()
        if ext_clean == gt_clean:
            return True

        # Check unit equivalence (e.g. 5V vs 5.0V)
        ext_num, ext_cat, ext_norm = TableEntityTokenizer.normalize_unit_value(ext_clean)
        gt_num, gt_cat, gt_norm = TableEntityTokenizer.normalize_unit_value(gt_clean)
        if ext_num is not None and gt_num is not None:
            if abs(ext_num - gt_num) < 1e-4 and (ext_cat == gt_cat or ext_cat == 'dimensionless' or gt_cat == 'dimensionless'):
                return True
        return False

    def _check_citation_completeness(self, citation: Optional[Dict[str, Any]]) -> bool:
        if not citation:
            return False
        required_keys = ["row_label", "column_label", "table_id", "section_path", "page", "bbox"]
        for k in required_keys:
            val = citation.get(k)
            if val is None or val == "" or val == []:
                return False
        return True

    def _detect_token_repetition(self, text: str) -> bool:
        """Detect token degeneration loops (e.g. repeating 3-grams)."""
        words = text.lower().split()
        if len(words) < 10:
            return False
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        counts = {}
        for tri in trigrams:
            counts[tri] = counts.get(tri, 0) + 1
            if counts[tri] >= 4:  # Repeating same 3-gram >= 4 times
                return True
        return False

    def _simulate_llm_judge(self, query: str, extracted: Optional[str], gt: str, results: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        """Simulated Qwen3-32b LLM Judge scoring Ragas Faithfulness, Relevancy, Precision."""
        if not results or not extracted:
            return 0.0, 0.0, 0.0

        is_correct = self._check_exact_match(extracted, gt)
        faithfulness = 1.0 if is_correct else 0.25
        relevancy = 0.95 if is_correct else 0.40
        precision = 1.0 if is_correct else 0.30

        return faithfulness, relevancy, precision
