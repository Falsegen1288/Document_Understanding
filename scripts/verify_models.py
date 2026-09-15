import os
import sys
import yaml
import shutil
import datetime
from pathlib import Path

os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "D:/sentence_transformers_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/huggingface_cache")

import torch


def verify_models():
    registry_path = Path("embedding_bench/registry/models.yaml")
    if not registry_path.exists():
        print(f"Registry config not found at {registry_path}")
        return

    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models_cfg = data.get("models", {})
    results = []

    # Disk usage check
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage("D:/")
        disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        try:
            usage = shutil.disk_usage("C:/")
            disk_free_gb = usage.free / (1024 ** 3)
        except Exception:
            disk_free_gb = -1.0

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    print("=" * 80)
    print("  PHASE 10 PREFLIGHT MODEL VERIFICATION")
    print(f"  CUDA Available: {cuda_available} | Device: {device_name}")
    print(f"  Drive D Free Space: {disk_free_gb:.2f} GB")
    print("=" * 80)

    for model_key, cfg in models_cfg.items():
        hf_id = cfg.get("hf_model_id", model_key)
        backend = cfg.get("backend_class", "Unknown")
        status = "PASS"
        err_msg = "-"

        print(f"\nChecking model '{model_key}' ({hf_id})...", flush=True)

        try:
            if "Vision" in backend:
                from transformers import AutoProcessor
                AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
            elif "MultiVector" in backend or "LocalST" in backend:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(hf_id)
            else:
                from transformers import AutoTokenizer
                AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        except Exception as e:
            status = "FAIL"
            err_msg = str(e).replace("\n", " ")[:150]
            print(f"  [FAIL] {model_key}: {err_msg}")
        else:
            print(f"  [PASS] {model_key} loaded successfully.")

        results.append({
            "model_key": model_key,
            "hf_id": hf_id,
            "backend": backend,
            "status": status,
            "error": err_msg
        })

    # Write results markdown report
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "phase10_model_preflight.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 10 Model Preflight Verification Report\n")
        f.write(f"**Timestamp**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- **CUDA Available**: `{cuda_available}` (`{device_name}`)\n")
        f.write(f"- **Drive Free Space**: `{disk_free_gb:.2f} GB`\n\n")
        f.write("| Model Key | HuggingFace ID | Backend Class | Status | Error Details |\n")
        f.write("| :--- | :--- | :--- | :---: | :--- |\n")

        for r in results:
            f.write(f"| `{r['model_key']}` | `{r['hf_id']}` | `{r['backend']}` | **{r['status']}** | {r['error']} |\n")

    print("\n" + "=" * 80)
    print(f"Preflight verification complete. Report written to {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    verify_models()
