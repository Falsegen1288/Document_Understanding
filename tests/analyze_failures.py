import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.sample_table_data import SAMPLE_TABLES
from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tests.eval_harness_tables import TableEvalHarness

def main():
    strategy_a = StrategyARowKVIndex()
    strategy_a.ingest_tables(SAMPLE_TABLES)

    harness = TableEvalHarness("data/table_qa_eval_dataset.json")

    print("=" * 80)
    print("DETAILED FAILURE ANALYSIS FOR STRATEGY A")
    print("=" * 80)

    failing_cases = []

    for qa in harness.qa_pairs:
        q_id = qa['query_id']
        q_type = qa['query_type']
        query = qa['query']
        gt_val = qa['ground_truth_value']
        gt_cit = qa['ground_truth_citation']
        
        results = strategy_a.search(query, top_k=5)
        
        top_val = results[0].get('extracted_value') if results else None
        top_cit = results[0].get('citation', {}) if results else None
        top_row = results[0].get('row_label') if results else None
        top_col = results[0].get('column_label') if results else None
        top_tbl = results[0].get('table_id') if results else None
        
        is_em = harness._check_exact_match(top_val, gt_val)
        
        if not is_em:
            failing_cases.append({
                "query_id": q_id,
                "query_type": q_type,
                "query": query,
                "gt_val": gt_val,
                "gt_citation": gt_cit,
                "top_val": top_val,
                "top_row": top_row,
                "top_col": top_col,
                "top_tbl": top_tbl,
                "top_citation": top_cit,
                "full_results": results
            })
            print(f"\n[FAIL] {q_id} ({q_type}): \"{query}\"")
            print(f"   Ground Truth: val=\"{gt_val}\" | row=\"{gt_cit['row_label']}\" | col=\"{gt_cit['column_label']}\" | tbl=\"{gt_cit['table_id']}\"")
            print(f"   Returned:     val=\"{top_val}\" | row=\"{top_row}\" | col=\"{top_col}\" | tbl=\"{top_tbl}\"")
            if results:
                print(f"   Row KV payload: {results[0].get('row_kv')}")
            else:
                print("   No results returned!")

    print("\n" + "=" * 80)
    print(f"Total Failing Cases: {len(failing_cases)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
