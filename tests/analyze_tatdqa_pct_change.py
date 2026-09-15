import os
import sys
import json
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arithmetic.symbolic_engine import SymbolicArithmeticEngine

def analyze_pct_change_failures():
    print("=" * 80, flush=True)
    print("  TASK 8.4: TAT-DQA PERCENTAGE_CHANGE FAILURE ANALYSIS", flush=True)
    print("=" * 80, flush=True)

    qa_path = "external_benchmarks/TAT-DQA/data/tatdqa_dataset_dev.json"
    if not os.path.exists(qa_path):
        print(f"Error: QA dataset not found at {qa_path}", flush=True)
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    engine = SymbolicArithmeticEngine()

    pct_queries = []
    for doc in data:
        questions = doc.get("questions", [])
        for q in questions:
            derivation = q.get("derivation", "")
            ans_type = q.get("answer_type", "")
            op = q.get("derivation_op", "")
            if op == "PERCENTAGE_CHANGE" or "CHANGE" in derivation.upper() or "%" in str(q.get("answer")):
                pct_queries.append((doc.get("table", {}), q))

    print(f"Found {len(pct_queries)} candidate PERCENTAGE_CHANGE / Change queries in dev set.", flush=True)

    sign_errors = 0
    formula_bugs = 0
    scale_mismatches = 0
    correct_evals = 0

    for idx, (tbl, q) in enumerate(pct_queries[:30]):
        query = q.get("question", "")
        gt_answer = q.get("answer")
        derivation = q.get("derivation", "")

        computed = engine.safe_eval_expression(derivation)
        is_match = engine.check_numeric_match(computed, gt_answer)

        # Check for sign reversal or multiplier mismatch (e.g. 0.15 vs 15%)
        is_sign_flip = False
        is_scale_100 = False
        if computed is not None and gt_answer is not None:
            try:
                c_val = float(computed)
                g_val = float(str(gt_answer).replace("%", "").strip())
                if abs(c_val + g_val) < 0.05:
                    is_sign_flip = True
                if abs(c_val * 100.0 - g_val) < 0.05 or abs(c_val / 100.0 - g_val) < 0.05:
                    is_scale_100 = True
            except ValueError:
                pass

        if is_match:
            correct_evals += 1
        elif is_sign_flip:
            sign_errors += 1
        elif is_scale_100:
            scale_mismatches += 1
        else:
            formula_bugs += 1

        print(f"\nQuery #{idx+1}: {query}", flush=True)
        print(f"  Derivation: '{derivation}' | GT Answer: {gt_answer}", flush=True)
        print(f"  Computed AST Value: {computed} | Match: {is_match}", flush=True)
        if is_sign_flip:
            print("  [DIAGNOSIS]: Sign Convention Error (Positive vs Negative Change)", flush=True)
        elif is_scale_100:
            print("  [DIAGNOSIS]: Scale Mismatch (Decimal vs Percentage Multiplier 100x)", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  TASK 8.4 SUMMARY FINDINGS", flush=True)
    print(f"  Total Analyzed Queries:   {len(pct_queries)}")
    print(f"  Correct Evals:           {correct_evals}")
    print(f"  Sign Convention Errors:   {sign_errors}")
    print(f"  Scale Mismatches (x100):  {scale_mismatches}")
    print(f"  Other Formula Failures:  {formula_bugs}")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    analyze_pct_change_failures()
