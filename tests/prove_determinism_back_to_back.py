import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.verify_phase9_forensic_fix import run_phase9_forensic_audit

def main():
    print("================================================================================", flush=True)
    print("  TASK 9.2: BACK-TO-BACK DETERMINISM PROOF RUN 1", flush=True)
    print("================================================================================", flush=True)
    run1 = run_phase9_forensic_audit()

    print("\n" + "=" * 80, flush=True)
    print("  TASK 9.2: BACK-TO-BACK DETERMINISM PROOF RUN 2 (ZERO CODE CHANGES)", flush=True)
    print("=" * 80, flush=True)
    run2 = run_phase9_forensic_audit()

    print("\n" + "=" * 80, flush=True)
    print("  TASK 9.2 DETERMINISM COMPARISON MATRIX", flush=True)
    print("=" * 80, flush=True)

    all_matched = True
    for domain in ["finance", "legal"]:
        r1 = run1[domain]
        r2 = run2[domain]

        ret_match = (r1["retrieval_acc"] == r2["retrieval_acc"])
        j_oracle_match = (r1["jaccard"]["oracle_hits"] == r2["jaccard"]["oracle_hits"])
        j_e2e_match = (r1["jaccard"]["e2e_hits"] == r2["jaccard"]["e2e_hits"])
        u_oracle_match = (r1["upgraded"]["oracle_hits"] == r2["upgraded"]["oracle_hits"])
        u_e2e_match = (r1["upgraded"]["e2e_hits"] == r2["upgraded"]["e2e_hits"])

        domain_matched = ret_match and j_oracle_match and j_e2e_match and u_oracle_match and u_e2e_match
        if not domain_matched:
            all_matched = False

        print(f"\n[{domain.upper()} DOMAIN REPRODUCIBILITY]:", flush=True)
        print(f"  Retrieval Acc:       Run1={r1['retrieval_acc']:.4f} | Run2={r2['retrieval_acc']:.4f} | Match: {ret_match}", flush=True)
        print(f"  Jaccard Oracle Acc:  Run1={r1['jaccard']['oracle_acc']:.4f} | Run2={r2['jaccard']['oracle_acc']:.4f} | Match: {j_oracle_match}", flush=True)
        print(f"  Jaccard True E2E:    Run1={r1['jaccard']['e2e_em']:.4f} | Run2={r2['jaccard']['e2e_em']:.4f} | Match: {j_e2e_match}", flush=True)
        print(f"  Upgraded Oracle Acc: Run1={r1['upgraded']['oracle_acc']:.4f} | Run2={r2['upgraded']['oracle_acc']:.4f} | Match: {u_oracle_match}", flush=True)
        print(f"  Upgraded True E2E:   Run1={r1['upgraded']['e2e_em']:.4f} | Run2={r2['upgraded']['e2e_em']:.4f} | Match: {u_e2e_match}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"  FINAL DETERMINISM PROOF VERDICT: [{'PASS' if all_matched else 'FAIL'}]", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
