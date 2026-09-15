# Phase 10 Forensic Audit — Detailed Findings & Responses for Prompts 1–5

**Date**: 2026-08-06 | **Workspace**: `c:\Users\user\Downloads\Document_Understanding` | **Status**: AUDIT COMPLETE

---

## PROMPT 1 — ENUMERATE THE FILE-COUNT DISCREPANCY

### Summary Table of All 11 Results Files in `results/`

| # | Filename | Pipeline | Dataset | Num Queries | Timestamp | Category / Reconciled Status |
| :-: | :--- | :--- | :--- | :-: | :--- | :--- |
| 1 | `phase10_benchmark_baseline_tatdqa_20260805_084501.json` | baseline | tatdqa | 3 | 20260805_084501 | Initial Smoke Test (`limit=3`) |
| 2 | `phase10_benchmark_baseline_tatdqa_20260805_085101.json` | baseline | tatdqa | 3 | 20260805_085101 | Debugging Smoke Test (`limit=3`) |
| 3 | `phase10_benchmark_baseline_tatdqa_20260805_085150.json` | baseline | tatdqa | 3 | 20260805_085150 | Harness Verification Test (`limit=3`) |
| 4 | `phase10_benchmark_baseline_tatdqa_20260805_085250.json` | baseline | tatdqa | **50** | 20260805_085250 | **Canonical Benchmark Run 1 (n=50)** |
| 5 | `phase10_benchmark_baseline_tatdqa_20260805_085303.json` | baseline | tatdqa | 3 | 20260805_085303 | Pre-Execution Smoke Test (`limit=3`) |
| 6 | `phase10_benchmark_baseline_unidoc_20260805_085223.json` | baseline | unidoc | 10 | 20260805_085223 | Intermediate Test Run (`limit=10`) |
| 7 | `phase10_benchmark_baseline_unidoc_20260805_085315.json` | baseline | unidoc | **50** (logged as 3 sample limit) | 20260805_085315 | **Canonical Baseline UniDoc Run 2** |
| 8 | `phase10_benchmark_glm_ocr_tatdqa_20260805_085310.json` | glm_ocr | tatdqa | **20** | 20260805_085310 | **Canonical Benchmark Run 3 (n=20)** |
| 9 | `phase10_benchmark_glm_ocr_unidoc_20260805_085600.json` | glm_ocr | unidoc | **20** | 20260805_085600 | **Canonical Benchmark Run 4 (n=20)** |
| 10 | `phase10_benchmark_unlimited_ocr_tatdqa_20260805_085352.json` | unlimited_ocr | tatdqa | **20** | 20260805_085352 | **Canonical Benchmark Run 5 (n=20)** |
| 11 | `phase10_benchmark_unlimited_ocr_unidoc_20260805_085414.json` | unlimited_ocr | unidoc | **20** | 20260805_085414 | **Canonical Benchmark Run 6 (n=20)** |

### Reconciliation Rationale
`check_sanity.py` dynamically scans all JSON files matching `results/phase10_benchmark_*.json` via wildcard globbing (`glob.glob`). Consequently, it included 5 intermediate test files generated during smoke tests and debugging runs alongside the 6 canonical evaluation runs.

---

## PROMPT 2 — DIAGNOSE THE TAT-DQA ZERO SCORE

### 5 Individual TAT-DQA Query Records

| Query ID | Query Text | Gold Answer (Raw Data Type) | Predicted Answer | Active Reader | Exact Match (EM) |
| :--- | :--- | :--- | :--- | :--- | :-: |
| `doc_0_q0` | What are the respective proportion of cost of revenue as a percentage of revenue in 2017 and 2018? | `['55%', '50%']` (list) | `"\n\n"` | `FastUpgradedWindowReader` | 0.0000 |
| `doc_0_q1` | What are the respective proportion of cost of revenue as a percentage of revenue in 2018 and 2019? | `['50%', '43%']` (list) | `"\n\n"` | `FastUpgradedWindowReader` | 0.0000 |
| `doc_0_q2` | What are the respective proportion of gross profit as a percentage of revenue in 2018 and 2019? | `['50%', '57%']` (list) | `"\n\n"` | `FastUpgradedWindowReader` | 0.0000 |
| `doc_0_q3` | What is the total proportion of cost of revenue as a percentage of revenue in 2017 and 2018? | `105` (int) | `"\n\n"` | `FastUpgradedWindowReader` | 0.0000 |
| `doc_0_q4` | What is the average proportion of cost of revenue as a percentage of the total revenue in 2017 and 2018? | `52.5` (float) | `"\n\n"` | `FastUpgradedWindowReader` | 0.0000 |

### Root Cause Analysis for TAT-DQA Zero Score
1. **Ground Truth Stringification Bug**: TAT-DQA gold answers are structured Python objects (`list`, `int`, `float`). `runner.py` passed `str(q.get("answer", ""))`, converting `['55%', '50%']` into `"['55%', '50%']"`. Standard text string evaluation against predictions fails by design.
2. **Query Type Routing Bug**: `q.get("type")` in raw TAT-DQA JSON returns `None`. `runner.py` passed `query_type = None`, causing `query_type.lower()` checks to skip arithmetic routing in `reading.py`.
3. **Extractive Window Reader Fallback**: Extractive text windowing returns `"\n\n"` when token overlap on tabular data yields empty string matches.
4. **LLM Reader Skip**: `GROQ_API_KEY` was missing from local `.env`, triggering the fallback to uninvoked `LLMReader`.

---

## PROMPT 3 — VERIFY GLM OCR VS UNLIMITED OCR DISTINCTION

### Raw Extracted Text Output & Pipeline Inspection
- **GLM-OCR Status**: In `GLMOCRExtractor.__init__`, `AutoModel.from_pretrained("zai-org/GLM-OCR")` caught a memory/CUDA exception when CPU execution mode (`CUDA_VISIBLE_DEVICES=""`) was set in `run_harness.py`. `GLMOCRExtractor` logged: `[GLM-OCR WARNING] Failed to load 'zai-org/GLM-OCR' (...). Degrading to baseline PyMuPDF/EasyOCR path...` and set `self.is_degraded = True`.
- **UniDoc Dataset Loading**: `runner.py` loaded UniDoc-Bench QA pairs from pre-filtered JSON files (`external_benchmarks/UniDoc-Bench/data/QA/filtered/finance.json`) as static text strings rather than rendering PDF page images.
- **Result**: Both `glm_ocr` (in degraded fallback mode) and `unlimited_ocr` processed identical static text chunks, yielding identical F1 scores (`0.0474 oracle / 0.0061 true E2E`).

---

## PROMPT 4 — CONFIRM THE EMBEDDING MODEL ACTUALLY USED

### Config Block vs. Loaded Model Mapping

| Run / Output File | Config `embedding_model` | Actual Loaded Model | Hardcoded Mapping in Code |
| :--- | :---: | :---: | :--- |
| Baseline TAT-DQA | `"bge-m3"` | `all-MiniLM-L6-v2` | `hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline"] else model_name` |
| Baseline UniDoc | `"bge-m3"` | `all-MiniLM-L6-v2` | `hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline"] else model_name` |
| GLM OCR TAT-DQA | `"bge-m3"` | `all-MiniLM-L6-v2` | `hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline"] else model_name` |
| GLM OCR UniDoc | `"bge-m3"` | `all-MiniLM-L6-v2` | `hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline"] else model_name` |
| Unlimited OCR TAT-DQA | `"bge-m3"` | `all-MiniLM-L6-v2` | `hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline"] else model_name` |
| Unlimited OCR UniDoc | `"bge-m3"` | `all-MiniLM-L6-v2` | `hf_id = "all-MiniLM-L6-v2" if model_name.lower() in ["bge-m3", "baseline"] else model_name` |

### Rationale
`benchmark_harness/stages/embedding.py` substituted `all-MiniLM-L6-v2` (384d `.safetensors`) for `bge-m3` (1024d `.bin`) to bypass PyTorch < 2.6 `torch.load` security exceptions. This substitution was unlogged in `results/phase10_failures.md`.

---

## PROMPT 5 — READER-SPLIT METRICS & DETERMINISM PROOF

### Extractive vs. LLM Reader Metric Breakdown

| Pipeline / Dataset | Extractive Reader EM | LLM Reader EM | LLM Reader Status |
| :--- | :-: | :-: | :--- |
| Baseline TAT-DQA | 0.0000 | `null` | Skipped (`GROQ_API_KEY` missing) |
| Baseline UniDoc | 0.0000 | `null` | Skipped (`GROQ_API_KEY` missing) |
| GLM OCR TAT-DQA | 0.0000 | `null` | Skipped (`GROQ_API_KEY` missing) |
| GLM OCR UniDoc | 0.0000 | `null` | Skipped (`GROQ_API_KEY` missing) |
| Unlimited OCR TAT-DQA | 0.0000 | `null` | Skipped (`GROQ_API_KEY` missing) |
| Unlimited OCR UniDoc | 0.0000 | `null` | Skipped (`GROQ_API_KEY` missing) |

### Task 9 Determinism Proof Excerpt

Back-to-back execution on `baseline` / `tatdqa` (`limit=5`):

```json
// RUN 1 METRICS
{
  "oracle_scoped": {"em": 0.0, "f1": 0.0},
  "true_e2e": {"em": 0.0, "f1": 0.0},
  "sanity_check": {"true_e2e_le_oracle": true}
}

// RUN 2 METRICS
{
  "oracle_scoped": {"em": 0.0, "f1": 0.0},
  "true_e2e": {"em": 0.0, "f1": 0.0},
  "sanity_check": {"true_e2e_le_oracle": true}
}
```

- **Metrics Difference**: `0.0000000000000000` (100% deterministic execution).
