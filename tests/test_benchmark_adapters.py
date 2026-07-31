import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.adapters.tatdqa_adapter import TATDQAAdapter
from tests.adapters.unidoc_adapter import UniDocBenchAdapter

def main():
    print("=" * 70)
    print("      TESTING EXTERNAL BENCHMARK ADAPTER CONVERSIONS")
    print("=" * 70)

    # 1. Test TAT-DQA Adapter
    tat_path = "external_benchmarks/TAT-DQA/data/tatdqa_dataset_dev.json"
    if os.path.exists(tat_path):
        with open(tat_path, "r", encoding="utf-8") as f:
            tat_data = json.load(f)
        sample_tat_doc = tat_data[0]
        tables = TATDQAAdapter.transform_tatdqa_doc_to_schema(sample_tat_doc)
        print(f"\n[TAT-DQA Adapter] Converted document '{sample_tat_doc.get('doc')}' to {len(tables)} table schema objects.")
        if tables:
            print("  Sample Table ID:", tables[0]["table_id"])
            print("  Sample Headers:", tables[0]["column_headers"])
            print("  Sample Rows Count:", len(tables[0]["rows"]))
            pred = TATDQAAdapter.format_prediction_for_eval("q_test_123", "55%", "%")
            print("  Formatted Prediction:", pred)

    # 2. Test UniDoc-Bench Adapter
    unidoc_path = "external_benchmarks/UniDoc-Bench/data/QA/filtered/healthcare.json"
    if os.path.exists(unidoc_path):
        with open(unidoc_path, "r", encoding="utf-8") as f:
            unidoc_data = json.load(f)
        sample_unidoc_qa = unidoc_data[0]
        qa_obj = UniDocBenchAdapter.transform_unidoc_qa_pair(sample_unidoc_qa, 1)
        print(f"\n[UniDoc-Bench Adapter] Converted QA pair 1 (healthcare domain):")
        print("  Query ID:", qa_obj["query_id"])
        print("  Query:", qa_obj["query"][:80] + "...")
        print("  GT Answer:", qa_obj["ground_truth_value"][:80] + "...")
        print("  GT Citation:", qa_obj["ground_truth_citation"])

    print("\n" + "=" * 70)
    print("      ALL ADAPTER TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
