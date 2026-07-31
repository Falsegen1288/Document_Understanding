import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.adapters.unidoc_adapter import UniDocBenchAdapter
from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tests.eval_harness_tables import TableEvalHarness

def evaluate_unidoc_domain(domain_file: str, index: StrategyARowKVIndex):
    domain_name = os.path.basename(domain_file).replace('.json', '')
    with open(domain_file, "r", encoding="utf-8") as f:
        unidoc_items = json.load(f)

    # Convert UniDoc QA items to harness format
    harness_dataset = []
    for idx, item in enumerate(unidoc_items):
        harness_dataset.append(UniDocBenchAdapter.transform_unidoc_qa_pair(item, idx))

    # Temporarily save transformed dataset to scratch path for harness
    scratch_path = f"data/unidoc_eval_{domain_name}.json"
    with open(scratch_path, "w", encoding="utf-8") as f:
        json.dump(harness_dataset, f, indent=2)

    # Evaluate via TableEvalHarness
    harness = TableEvalHarness(scratch_path)
    metrics = harness.evaluate_index(index, f"UniDoc-Bench ({domain_name})")

    # Clean up scratch file
    if os.path.exists(scratch_path):
        os.remove(scratch_path)

    return domain_name, metrics

def main():
    domains = ["healthcare", "finance", "legal"]
    base_dir = os.path.join("external_benchmarks", "UniDoc-Bench", "data", "QA", "filtered")

    print("=" * 80)
    print("      RUNNING UNIDOC-BENCH EVALUATION ON HEALTHCARE, FINANCE, LEGAL")
    print("=" * 80)

    # Initialize Strategy A
    index = StrategyARowKVIndex()

    domain_results = {}
    for d_name in domains:
        d_path = os.path.join(base_dir, f"{d_name}.json")
        if os.path.exists(d_path):
            d_name_out, metrics = evaluate_unidoc_domain(d_path, index)
            domain_results[d_name_out] = metrics

    print("\n" + "=" * 80)
    print(f"{'Domain':<20} | {'Cell EM Acc':<14} | {'Ragas Faithfulness':<18} | {'Ragas Relevancy':<16} | {'Chunk Citation CCR':<18} | {'Latency':<8}")
    print("-" * 80)
    for d_name, m in domain_results.items():
        print(f"{d_name:<20} | {m['cell_exact_match_accuracy']:<14.2%} | {m['ragas_faithfulness']:<18.3f} | {m['ragas_answer_relevancy']:<16.3f} | {m['citation_completeness_rate']:<18.2%} | {m['avg_latency_ms']:<8.2f} ms")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
