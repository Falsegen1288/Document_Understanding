import glob
import json
import sys
from pathlib import Path


def check_metric_sanity():
    json_files = glob.glob("results/phase10_benchmark_*.json")
    if not json_files:
        print("No phase10_benchmark_*.json result files found in results/.")
        return True

    all_passed = True
    print("=" * 80)
    print("  PHASE 10 METRIC SANITY CHECK (True_E2E <= Oracle_Scoped)")
    print("=" * 80)

    for filepath in json_files:
        filename = Path(filepath).name
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            sanity = data.get("sanity_check", {})
            passed = sanity.get("true_e2e_le_oracle", False)

            metrics = data.get("metrics", {})
            oracle_em = metrics.get("oracle_scoped", {}).get("em", 0.0)
            e2e_em = metrics.get("true_e2e", {}).get("em", 0.0)

            status_str = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"| {filename:<55} | Status: {status_str} | Oracle EM: {oracle_em:.4f} | True E2E EM: {e2e_em:.4f} |")
        except Exception as e:
            print(f"| {filename:<55} | Status: ERROR ({e}) |")
            all_passed = False

    print("=" * 80)
    print(f"Sanity Check Overall Status: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = check_metric_sanity()
    if not success:
        sys.exit(1)
