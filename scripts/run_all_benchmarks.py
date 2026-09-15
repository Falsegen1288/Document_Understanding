import sys
import os
import subprocess
import json
import glob

PIPELINES = ["baseline", "glm_ocr", "unlimited_ocr"]
DATASETS = ["unidoc", "tatdqa"]

def run_single(pipeline, dataset, limit=None, use_pal=True):
    print("\n" + "=" * 80)
    print(f"STARTING BENCHMARK: Pipeline={pipeline} | Dataset={dataset} | Limit={limit} | PAL={use_pal}")
    print("=" * 80, flush=True)

    cmd = [
        r".venv_stable\Scripts\python.exe",
        "run_harness.py",
        "--pipeline", pipeline,
        "--dataset", dataset
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    if use_pal:
        cmd.append("--use-pal")

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    res = subprocess.run(cmd, env=env)
    return res.returncode

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Query limit per benchmark")
    parser.add_argument("--no-pal", action="store_true", help="Disable PAL arithmetic")
    args = parser.parse_args()

    results_summary = []

    for dataset in DATASETS:
        for pipeline in PIPELINES:
            rc = run_single(pipeline, dataset, limit=args.limit, use_pal=not args.no_pal)
            if rc != 0:
                print(f"FAILED: {pipeline} on {dataset}")

    print("\n" + "=" * 80)
    print("ALL RUNS COMPLETE — SUMMARY OF LATEST RESULTS")
    print("=" * 80)
    from benchmark_harness.stages.scorer_reweight import recompute_headline_from_existing_run
    
    for dataset in DATASETS:
        for pipeline in PIPELINES:
            pattern = f"results/phase10_benchmark_{pipeline}_{dataset}_*.json"
            matched = sorted(glob.glob(pattern), reverse=True)
            if matched:
                latest = matched[0]
                with open(latest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                m = data.get("metrics", {}).get("true_e2e", {})
                ret = data.get("metrics", {}).get("retrieval_layer", {})
                h_score = m.get("headline_score", 0.0)
                hit1 = ret.get("hit@1", 0.0)
                num_em = m.get("numeric_em", 0.0)
                f1 = m.get("f1", 0.0)
                print(f"{pipeline:15s} | {dataset:8s} | Hit@1: {hit1:.4f} | Headline: {h_score:.4f} | NumEM: {num_em:.4f} | F1: {f1:.4f} | File: {os.path.basename(latest)}")

if __name__ == "__main__":
    main()
