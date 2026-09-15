import os
import sys
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.adapters.unidoc_adapter import UniDocBenchAdapter
from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tests.eval_harness_tables import TableEvalHarness

def main():
    print("=" * 80)
    print("      PART B: UNIDOC-BENCH EQUIVALENT SCRUTINY PASS & CITATION AUDIT")
    print("=" * 80)

    qa_dir = os.path.join("external_benchmarks", "UniDoc-Bench", "data", "QA", "filtered")
    domains = ["healthcare", "finance", "legal"]

    domain_audit_results = {}

    for domain in domains:
        qa_path = os.path.join(qa_dir, f"{domain}.json")
        if not os.path.exists(qa_path):
            print(f"Skipping {domain}: {qa_path} missing.")
            continue

        with open(qa_path, "r", encoding="utf-8") as f:
            raw_qa = json.load(f)

        text_table_qas = []
        figure_qas = []

        for idx, item in enumerate(raw_qa):
            ans_type = item.get("answer_type", "")
            qa_obj = UniDocBenchAdapter.transform_unidoc_qa_pair(item, idx)
            if ans_type == "image_only" or "figure" in item.get("question_type", ""):
                figure_qas.append(item)
            else:
                text_table_qas.append((item, qa_obj))

        print(f"\n[{domain.upper()}] Total: {len(raw_qa)} | Text/Table: {len(text_table_qas)} | Figure (Not Supported): {len(figure_qas)}")

        # Audit 5 sample QA pairs for source file citation provenance matching
        sample_traces = []
        for idx, (raw_item, qa_obj) in enumerate(text_table_qas[:5]):
            gt_ans = qa_obj.get("ground_truth", "")
            evidence_files = raw_item.get("evidence", [])
            doc_id = raw_item.get("doc_id", "")
            
            sample_traces.append({
                "query_id": f"{domain}_{idx}",
                "question": qa_obj.get("question"),
                "gt_ans": gt_ans,
                "doc_id": doc_id,
                "evidence_files": evidence_files,
            })

        domain_audit_results[domain] = {
            "total": len(raw_qa),
            "text_table_count": len(text_table_qas),
            "figure_count": len(figure_qas),
            "samples": sample_traces
        }

    print("\n" + "=" * 80)
    print("UNIDOC-BENCH CITATION PROVENANCE AUDIT SAMPLE TRACES")
    print("=" * 80)
    for dom, res in domain_audit_results.items():
        print(f"\n--- {dom.upper()} DOMAIN SAMPLE TRACES ---")
        for s in res["samples"]:
            print(f"Query: \"{s['question']}\"")
            print(f"  Doc ID: {s['doc_id']} | Evidence Files: {s['evidence_files']}")
            print(f"  GT Answer: {s['gt_ans']}")

if __name__ == "__main__":
    main()
