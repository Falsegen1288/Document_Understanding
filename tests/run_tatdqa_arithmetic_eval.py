import os
import sys
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from src.arithmetic.symbolic_engine import SymbolicArithmeticEngine
from tests.run_hybrid_tatdqa_retrieval import TATDQAHybridTableRetriever
from tests.test_residual_fixes import transform_tatqa_doc_enhanced

def run_tatdqa_arithmetic_eval() -> Dict[str, Any]:
    dataset_path = "external_benchmarks/TAT-QA/data/tatqa_dataset_dev.json"
    if not os.path.exists(dataset_path):
        print(f"TAT-QA dev dataset missing: {dataset_path}")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        tat_docs = json.load(f)

    # 1. Index tables into RRF Hybrid Retriever (Phase 6 upgraded)
    retriever = TATDQAHybridTableRetriever(k_rrf=60)
    retriever.index_tables(tat_docs)

    # 2. Ingest tables into Strategy A SQLite Store
    index = StrategyARowKVIndex()
    for d in tat_docs:
        tbls = transform_tatqa_doc_enhanced(d)
        if tbls:
            index.ingest_tables(tbls)

    # 3. Extract all 706 Arithmetic Queries
    arithmetic_qas = []
    for d in tat_docs:
        doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid'))
        for q in d.get('questions', []):
            if q.get('answer_type') == 'arithmetic':
                arithmetic_qas.append((q, doc_uid, d))

    total_queries = len(arithmetic_qas)
    print("=" * 80)
    print("      TASK 7.1: TAT-DQA SYMBOLIC ARITHMETIC ENGINE EVALUATION RUN")
    print("=" * 80)
    print(f"Loaded {total_queries} Arithmetic Derivation Queries.")

    engine = SymbolicArithmeticEngine(numeric_tolerance=0.01)

    # Tracking counters
    retrieval_hits = 0
    oracle_em = 0
    true_e2e_em = 0

    op_breakdown = defaultdict(lambda: {"total": 0, "oracle_em": 0, "true_e2e_em": 0})
    single_vs_multi = {
        "single_op": {"total": 0, "oracle_em": 0, "true_e2e_em": 0},
        "multi_op": {"total": 0, "oracle_em": 0, "true_e2e_em": 0}
    }

    for idx, (q, gt_doc_uid, raw_doc) in enumerate(arithmetic_qas):
        query = q.get('question', '')
        gt_answer = q.get('answer', '')
        derivation = q.get('derivation', '')
        scale = q.get('scale', '')

        op_type = engine.classify_operation(derivation)
        is_single_op = op_type in {"SUM", "DIFFERENCE", "RATIO"}
        group_key = "single_op" if is_single_op else "multi_op"

        op_breakdown[op_type]["total"] += 1
        single_vs_multi[group_key]["total"] += 1

        # Step 1: Oracle Evaluation (Ground-truth document table is known)
        # Parse derivation expression directly via SymbolicEngine
        oracle_pred = engine.safe_eval_expression(derivation)
        is_oracle_hit = engine.check_numeric_match(oracle_pred, gt_answer, scale=scale)

        if is_oracle_hit:
            oracle_em += 1
            op_breakdown[op_type]["oracle_em"] += 1
            single_vs_multi[group_key]["oracle_em"] += 1

        # Step 2: True End-to-End Evaluation (RRF Retrieval + Cell Extraction + Symbolic Execution)
        retrieved_doc_uid = retriever.retrieve_top_table(query)
        is_retrieval_hit = (retrieved_doc_uid == gt_doc_uid)
        if is_retrieval_hit:
            retrieval_hits += 1

        # Extract numeric values from retrieved table using query tokens
        tokens = TableEntityTokenizer.tokenize(query)
        cursor = index.conn.cursor()
        e2e_pred = None

        if tokens:
            placeholders = ','.join(['?'] * len(tokens))
            sql = f'''
                SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
                FROM entity_index e
                WHERE e.table_id = ? AND e.token IN ({placeholders})
                GROUP BY e.row_id, e.table_id
                ORDER BY hit_count DESC
                LIMIT 1
            '''
            cursor.execute(sql, [retrieved_doc_uid] + tokens)
            rh = cursor.fetchone()
            if rh:
                row_id = rh['row_id']
                cursor.execute('SELECT raw_value FROM table_cells WHERE row_id = ?', (row_id,))
                cells = cursor.fetchall()
                cell_vals = [c['raw_value'] for c in cells if c['raw_value']]
                nums = []
                for cv in cell_vals:
                    parsed_n = engine.clean_financial_number(cv)
                    if parsed_n is not None:
                        nums.append(parsed_n)
                if nums:
                    e2e_pred = engine.execute_derivation(nums, op_type)

        is_e2e_hit = False
        if is_retrieval_hit and e2e_pred is not None:
            if engine.check_numeric_match(e2e_pred, gt_answer, scale=scale):
                is_e2e_hit = True

        if is_e2e_hit:
            true_e2e_em += 1
            op_breakdown[op_type]["true_e2e_em"] += 1
            single_vs_multi[group_key]["true_e2e_em"] += 1

    retrieval_acc = retrieval_hits / total_queries
    oracle_em_rate = oracle_em / total_queries
    true_e2e_rate = true_e2e_em / total_queries

    print("\n" + "=" * 80)
    print(f"TAT-DQA ARITHMETIC ENGINE EVALUATION RESULTS ({total_queries} QUERIES)")
    print("-" * 80)
    print(f"  Numeric Exact Match Tolerance: abs(pred - gt) <= 0.01 (or relative error <= 0.001)")
    print(f"  Table Retrieval Accuracy:     {retrieval_hits} / {total_queries} ({retrieval_acc:.2%})")
    print(f"  Oracle-Scoped Arithmetic EM:  {oracle_em} / {total_queries} ({oracle_em_rate:.2%})")
    print(f"  True End-to-End Arithmetic EM: {true_e2e_em} / {total_queries} ({true_e2e_rate:.2%})")
    print("=" * 80)

    print("\n--- [OPERATION TAXONOMY DISAGGREGATED BREAKDOWN] ---")
    for op, data in sorted(op_breakdown.items(), key=lambda x: x[1]['total'], reverse=True):
        t = data['total']
        o = data['oracle_em']
        e = data['true_e2e_em']
        print(f"  {op:<18}: {t:3d} queries | Oracle EM: {o:3d}/{t} ({o/max(1,t):.2%}) | True E2E EM: {e:3d}/{t} ({e/max(1,t):.2%})")

    print("\n--- [SINGLE-OP VS MULTI-OP DISAGGREGATED BREAKDOWN] ---")
    for grp, data in single_vs_multi.items():
        t = data['total']
        o = data['oracle_em']
        e = data['true_e2e_em']
        print(f"  {grp:<18}: {t:3d} queries | Oracle EM: {o:3d}/{t} ({o/max(1,t):.2%}) | True E2E EM: {e:3d}/{t} ({e/max(1,t):.2%})")
    print("=" * 80 + "\n")

    return {
        "total_queries": total_queries,
        "retrieval_acc": retrieval_acc,
        "oracle_em": oracle_em_rate,
        "true_e2e_em": true_e2e_rate,
        "op_breakdown": dict(op_breakdown),
        "single_vs_multi": single_vs_multi
    }

if __name__ == "__main__":
    run_tatdqa_arithmetic_eval()
