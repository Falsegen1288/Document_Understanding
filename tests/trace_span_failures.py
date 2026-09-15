import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external_benchmarks", "TAT-QA")))

from tests.adapters.tatdqa_adapter import TATDQAAdapter
from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tatqa_eval import evaluate_json

def main():
    with open("external_benchmarks/TAT-QA/data/tatqa_dataset_dev.json", "r", encoding="utf-8") as f:
        tat_docs = json.load(f)

    # Initialize Strategy A and ingest all 274 tables
    index = StrategyARowKVIndex()
    doc_table_map = {}
    for d in tat_docs:
        tbls = TATDQAAdapter.transform_tatdqa_doc_to_schema(d)
        if tbls:
            index.ingest_tables(tbls)
            doc_uid = tbls[0]['table_id']
            doc_table_map[doc_uid] = tbls[0]

    # Isolate table-only span queries (answer_from == 'table' and answer_type == 'span')
    table_only_span_qas = []
    for d in tat_docs:
        doc_uid = d.get('table', {}).get('uid') or d.get('doc', {}).get('uid')
        for q in d.get('questions', []):
            if q.get('answer_from') == 'table' and q.get('answer_type') == 'span':
                table_only_span_qas.append((q, d, doc_uid))

    print(f"Total Table-Only Span Queries: {len(table_only_span_qas)}")

    # Audit 15 failing queries
    failing_traces = []
    for q, d, doc_uid in table_only_span_qas:
        query = q['question']
        gt = q['answer']
        q_uid = q['uid']

        # Global search (un-scoped)
        global_results = index.search(query, top_k=3)
        top_global = global_results[0] if global_results else {}

        # Check if match
        is_em = False
        if top_global.get('extracted_value'):
            val_clean = str(top_global['extracted_value']).strip().lower()
            if isinstance(gt, list):
                is_em = val_clean in [str(g).strip().lower() for g in gt]
            else:
                is_em = val_clean == str(gt).strip().lower()

        if not is_em and len(failing_traces) < 15:
            failing_traces.append({
                "q_uid": q_uid,
                "query": query,
                "gt": gt,
                "gt_doc_uid": str(doc_uid),
                "top_global_table": top_global.get('table_id'),
                "top_global_row": top_global.get('row_label'),
                "top_global_col": top_global.get('column_label'),
                "top_global_val": top_global.get('extracted_value'),
                "results": global_results,
                "raw_doc_table": d.get('table', {}).get('table')
            })

    print("\n" + "=" * 80)
    print("STEP 1 & 2: 15 DETAILED TABLE-ONLY SPAN QUERY FAILING TRACES")
    print("=" * 80)

    for idx, tr in enumerate(failing_traces):
        print(f"\n[Trace #{idx+1}] Q-UID: {tr['q_uid']} | Document UID: {tr['gt_doc_uid']}")
        print(f"  Query Text:          \"{tr['query']}\"")
        print(f"  Ground Truth Answer: {tr['gt']}")
        print(f"  Matched Table ID:    {tr['top_global_table']} (Expected: {tr['gt_doc_uid']})")
        print(f"  Matched Row Label:   {tr['top_global_row']}")
        print(f"  Matched Column Label:{tr['top_global_col']}")
        print(f"  Matched Value:       \"{tr['top_global_val']}\"")
        print(f"  Raw Source Table Grid (First 3 rows): {tr['raw_doc_table'][:3] if tr['raw_doc_table'] else None}")
        print("-" * 80)

if __name__ == "__main__":
    main()
