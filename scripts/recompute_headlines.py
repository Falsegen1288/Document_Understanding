import glob
import json
import os
from benchmark_harness.stages.scorer_reweight import recompute_headline_from_existing_run

def main():
    files = glob.glob("results/*.json")
    print(f"Checking {len(files)} result files in results/...")
    
    valid_count = 0
    for path in sorted(files, reverse=True)[:10]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = data.get("metrics", {}).get("true_e2e") or data.get("true_e2e")
            if m:
                bundle = recompute_headline_from_existing_run(m)
                headline = bundle["headline_score"]
                f1 = m.get("f1", m.get("token_f1", 0.0))
                cont = m.get("containment", m.get("answer_containment", 0.0))
                pipe = data.get("pipeline", "unknown")
                dataset = data.get("dataset", "unknown")
                print(f"[{pipe}/{dataset}] {os.path.basename(path)} -> Headline: {headline:.4f} (F1: {f1:.4f}, Cont: {cont:.4f})")
                valid_count += 1
        except Exception as e:
            print(f"Error on {path}: {e}")
            
    print(f"Successfully recomputed {valid_count} recent runs.")

if __name__ == "__main__":
    main()
