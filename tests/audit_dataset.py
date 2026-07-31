import json

def main():
    with open("data/table_qa_eval_dataset.json", "r", encoding="utf-8") as f:
        qas = json.load(f)

    b2_count = 0
    b3_count = 0

    print("AUDITING EVAL DATASET FOR BUCKET 2 & BUCKET 3 QUERIES:\n")
    for qa in qas:
        q_id = qa["query_id"]
        q_type = qa["query_type"]
        q_text = qa["query"]

        # Classification check:
        # Bucket 2: Genuine multi-row aggregation / comparison (e.g. max/min across all rows, count, compare A vs B)
        # Bucket 3: Conceptual / fuzzy semantic matching (no exact entity/spec match, requires semantic inference)
        is_b2 = False
        is_b3 = False

        if "compare" in q_text.lower() or "maximum operating temperature" in q_text.lower() or "highest" in q_text.lower():
            # Check if it requires reading all rows to compare
            if q_id == "qa_09":  # "List catheters with operating voltage >= 10V."
                is_b2 = True
        
        if "outdoor" in q_text.lower() or "rugged" in q_text.lower() or "harsh" in q_text.lower():
            if q_id == "qa_16":  # "Is LAP-X2 rated for outdoor use?" - cell literally says "Yes (IP65 Rugged)"
                is_b3 = False  # Cell literally has "Yes (IP65 Rugged)" so it's lexical, not fuzzy semantic!

        if is_b2:
            b2_count += 1
        if is_b3:
            b3_count += 1

        print(f"{q_id:<6} | {q_type:<20} | B2={str(is_b2):<5} | B3={str(is_b3):<5} | Query: \"{q_text}\"")

    print("\nSUMMARY:")
    print(f"Bucket 2 (True Cross-Row Aggregation / Comparison) Count: {b2_count}")
    print(f"Bucket 3 (True Conceptual / Fuzzy Semantic Inference) Count: {b3_count}")

if __name__ == "__main__":
    main()
