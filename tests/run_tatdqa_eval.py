import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external_benchmarks", "TAT-QA")))

from tests.adapters.tatdqa_adapter import TATDQAAdapter
from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tatqa_eval import evaluate_json

def main():
    gold_path = os.path.join("external_benchmarks", "TAT-QA", "data", "tatqa_dataset_dev.json")
    if not os.path.exists(gold_path):
        print(f"Error: Gold dataset not found at {gold_path}")
        return

    print("=" * 80)
    print("      RUNNING REAL TAT-DQA BENCHMARK EVALUATION (OFFICIAL EVAL SCRIPT)")
    print("=" * 80)

    with open(gold_path, "r", encoding="utf-8") as f:
        golden_docs = json.load(f)

    # Initialize Strategy A
    index = StrategyARowKVIndex()
    
    # Ingest document tables into Strategy A
    total_tables_ingested = 0
    for doc in golden_docs:
        tables = TATDQAAdapter.transform_tatdqa_doc_to_schema(doc)
        if tables:
            index.ingest_tables(tables)
            total_tables_ingested += len(tables)

    print(f"Ingested {total_tables_ingested} tables from {len(golden_docs)} TAT-DQA documents into Strategy A store.")

    # Execute predictions for all QA pairs
    predicted_answers = {}
    total_qa_pairs = 0
    start_time = time.time()

    for doc in golden_docs:
        for qa in doc.get("questions", []):
            q_id = qa["uid"]
            query = qa["question"]
            total_qa_pairs += 1
            
            # Execute Strategy A lookup
            results = index.search(query, top_k=3)
            
            pred_val = ""
            pred_scale = ""
            if results:
                pred_val = results[0].get("extracted_value", "")
            
            predicted_answers[q_id] = [pred_val, pred_scale]

    elapsed = time.time() - start_time
    print(f"Executed {total_qa_pairs} queries in {elapsed:.2f} seconds ({elapsed/total_qa_pairs*1000:.2f} ms/query).")

    print("\n--- OFFICIAL TAT-QA / TAT-DQA EVALUATION METRICS ---")
    evaluate_json(golden_docs, predicted_answers)
    print("=" * 80)

if __name__ == "__main__":
    main()
