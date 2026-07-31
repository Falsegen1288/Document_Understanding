import json
import os
import sys

# Ensure root src directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.sample_table_data import SAMPLE_TABLES
from src.table_indexing.flattened_baseline import FlattenedTableBaselineIndex
from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tests.eval_harness_tables import TableEvalHarness

def main():
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "table_qa_eval_dataset.json"))
    
    print("=" * 70)
    print("      RUNNING PHASE 2 BENCHMARK: FLATTENED BASELINE vs STRATEGY A")
    print("=" * 70)
    
    # 1. Initialize and ingest into Flattened Baseline
    baseline = FlattenedTableBaselineIndex()
    baseline.ingest_tables(SAMPLE_TABLES)
    
    # 2. Initialize and ingest into Strategy A
    strategy_a = StrategyARowKVIndex()
    strategy_a.ingest_tables(SAMPLE_TABLES)
    
    # 3. Evaluate Baseline
    harness = TableEvalHarness(dataset_path)
    baseline_metrics = harness.evaluate_index(baseline, "Flattened Text Chunk Baseline")
    
    # 4. Evaluate Strategy A
    strategy_a_metrics = harness.evaluate_index(strategy_a, "Strategy A (Structured Row KV Index)")
    
    # 5. Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Metric':<35} | {'Flattened Baseline':<20} | {'Strategy A':<15}")
    print("-" * 70)
    print(f"{'Cell-Level Exact Match Acc':<35} | {baseline_metrics['cell_exact_match_accuracy']:<20.2%} | {strategy_a_metrics['cell_exact_match_accuracy']:<15.2%}")
    print(f"{'Spec/Cell Hit Rate@5':<35} | {baseline_metrics['spec_hit_rate_at_5']:<20.2%} | {strategy_a_metrics['spec_hit_rate_at_5']:<15.2%}")
    print(f"{'Citation Completeness Rate':<35} | {baseline_metrics['citation_completeness_rate']:<20.2%} | {strategy_a_metrics['citation_completeness_rate']:<15.2%}")
    print(f"{'Ragas Faithfulness':<35} | {baseline_metrics['ragas_faithfulness']:<20.3f} | {strategy_a_metrics['ragas_faithfulness']:<15.3f}")
    print(f"{'Ragas Answer Relevancy':<35} | {baseline_metrics['ragas_answer_relevancy']:<20.3f} | {strategy_a_metrics['ragas_answer_relevancy']:<15.3f}")
    print(f"{'Avg Query Latency (ms)':<35} | {baseline_metrics['avg_latency_ms']:<20.2f} | {strategy_a_metrics['avg_latency_ms']:<15.2f}")
    print(f"{'Token Repetition Rate':<35} | {baseline_metrics['token_degeneration_rate']:<20.2%} | {strategy_a_metrics['token_degeneration_rate']:<15.2%}")
    print("=" * 70 + "\n")
    
    # Return metrics dictionary
    return {
        "baseline": baseline_metrics,
        "strategy_a": strategy_a_metrics
    }

if __name__ == "__main__":
    main()
