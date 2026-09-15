# Phase 9 Report: Forensic Audit, Determinism Proof & Corrected Reader Re-Validation

**Author:** Senior ML/RAG Systems Engineer  
**Project:** Document Understanding  
**Date:** August 4, 2026  
**Status:** **PASSED & VERIFIED** (Invariant `True_E2E <= Oracle` strictly enforced; Back-to-Back Determinism proven)

---

## 1. Executive Summary & Root Cause Forensic Audit (Task 9.1)

In Phase 8 Task 8.3, an empirical anomaly was detected where reported **True End-to-End Accuracy exceeded Oracle-Scoped Accuracy** (Finance E2E 12.02% vs Oracle 4.81%; Legal E2E 8.93% vs Oracle 3.57%). Per strict ML/RAG metric hygiene, `True_E2E <= Oracle_Scoped` is a non-negotiable physical invariant because True E2E requires *both* correct retrieval and correct extraction, whereas Oracle gives ground-truth retrieval for free.

### A. Ground-Truth Root Cause Identified
UniDoc-Bench PDFs are multi-page documents partitioned into page-level chunks indexed as `doc_id_chunk_0`, `doc_id_chunk_1`, `doc_id_chunk_2`, etc.
1. **The Phase 8 Oracle Evaluator Defect**: When evaluating Oracle-Scoped accuracy, the Phase 8 evaluator performed a single string prefix key lookup (`gt_doc_ids[0] in doc_text_map`), which defaulted to retrieving **only `chunk_0` text**. If ground-truth answer text resided on page 3 (`chunk_2`), Oracle evaluated `chunk_0`, missed the answer, and scored **0**.
2. **The Phase 8 E2E Evaluator Behavior**: In E2E mode, RRF hybrid retrieval correctly identified `chunk_2` as a top candidate. The reader searched `chunk_2`, extracted the answer, and scored **1**.
3. **Result**: For queries where ground-truth answers were on pages > 1, E2E scored **1** while Oracle scored **0**, causing `E2E > Oracle`. Additionally, attempting to concatenate all pages of a 50-page document into a single mega-string diluted sliding window scores with distant noise.

### B. Forensic Fix Implemented
In `tests/verify_phase9_forensic_fix.py`, Oracle evaluation was refactored to `get_all_gt_chunks()`. This iterates over **every ground-truth document chunk individually**, evaluating reader accuracy across each chunk at the exact same page/chunk-level granularity as E2E retrieval.

---

## 2. Back-to-Back Pipeline Determinism Proof (Task 9.2)

To satisfy Section 0.2 mandates, `tests/prove_determinism_back_to_back.py` executed `tests/verify_phase9_forensic_fix.py` **twice in a row, back-to-back, with zero code changes**.

### Reproducibility Comparison Matrix (Run 1 vs Run 2)

| Domain | Evaluation Metric | Run 1 Value | Run 2 Value | Digit Match Status |
| :--- | :--- | :---: | :---: | :---: |
| **Finance** | Retrieval Accuracy (RRF) | **51.92% (108 / 208)** | **51.92% (108 / 208)** | **MATCH (True)** |
| **Finance** | Jaccard Oracle Scoped Acc | **5.77% (12 / 208)** | **5.77% (12 / 208)** | **MATCH (True)** |
| **Finance** | Jaccard True E2E Answer EM | **1.92% (4 / 208)** | **1.92% (4 / 208)** | **MATCH (True)** |
| **Finance** | Upgraded Window Oracle Scoped Acc | **17.79% (37 / 208)** | **17.79% (37 / 208)** | **MATCH (True)** |
| **Finance** | Upgraded Window True E2E Answer EM | **12.02% (25 / 208)** | **12.02% (25 / 208)** | **MATCH (True)** |
| **Legal** | Retrieval Accuracy (RRF) | **55.36% (124 / 224)** | **55.36% (124 / 224)** | **MATCH (True)** |
| **Legal** | Jaccard Oracle Scoped Acc | **8.48% (19 / 224)** | **8.48% (19 / 224)** | **MATCH (True)** |
| **Legal** | Jaccard True E2E Answer EM | **4.46% (10 / 224)** | **4.46% (10 / 224)** | **MATCH (True)** |
| **Legal** | Upgraded Window Oracle Scoped Acc | **12.05% (27 / 224)** | **12.05% (27 / 224)** | **MATCH (True)** |
| **Legal** | Upgraded Window True E2E Answer EM | **8.93% (20 / 224)** | **8.93% (20 / 224)** | **MATCH (True)** |

**FINAL DETERMINISM PROOF VERDICT**: **`[PASS]`** (10/10 metrics match digit-for-digit across back-to-back runs).

---

## 3. Corrected Reader Re-Validation & Performance Deltas (Task 9.3)

With chunk-level Oracle evaluation restored and metric hygiene enforced (`True_E2E <= Oracle`), we re-evaluate the actual impact of `FastUpgradedWindowReader` over baseline `JaccardReader`.

### Corrected Dual-Domain Metrics Summary

```
================================================================================
  CORRECTED PHASE 9 DUAL-DOMAIN READER RE-EVALUATION MATRIX
================================================================================

1. FINANCE DOMAIN (208 QA Pairs):
   - Retrieval Accuracy (Dense RRF): 108 / 208 (51.92%)
   -----------------------------------------------------------------------------
   - Baseline Jaccard Reader:
     * Oracle-Scoped Answer Acc: 12 / 208 (5.77%)
     * True End-to-End Answer EM:  4 / 208 (1.92%)
     * Sanity Check (True_E2E <= Oracle): [PASS]
   -----------------------------------------------------------------------------
   - FastUpgradedWindowReader (Sliding 3-Sentence Window):
     * Oracle-Scoped Answer Acc: 37 / 208 (17.79%)  [+12.02% gain over Jaccard]
     * True End-to-End Answer EM: 25 / 208 (12.02%)  [+10.10% gain over Jaccard]
     * Sanity Check (True_E2E <= Oracle): [PASS]

2. LEGAL DOMAIN (224 QA Pairs):
   - Retrieval Accuracy (Dense RRF): 124 / 224 (55.36%)
   -----------------------------------------------------------------------------
   - Baseline Jaccard Reader:
     * Oracle-Scoped Answer Acc: 19 / 224 (8.48%)
     * True End-to-End Answer EM: 10 / 224 (4.46%)
     * Sanity Check (True_E2E <= Oracle): [PASS]
   -----------------------------------------------------------------------------
   - FastUpgradedWindowReader (Sliding 3-Sentence Window):
     * Oracle-Scoped Answer Acc: 27 / 224 (12.05%)  [+3.57% gain over Jaccard]
     * True End-to-End Answer EM: 20 / 224 (8.93%)   [+4.47% gain over Jaccard]
     * Sanity Check (True_E2E <= Oracle): [PASS]
================================================================================
```

### Key Takeaways
1. **Mathematical Invariant Holds**: `True_E2E <= Oracle_Scoped` holds strictly across all domain and reader configurations (Finance Oracle 17.79% >= E2E 12.02%; Legal Oracle 12.05% >= E2E 8.93%).
2. **Reader Upgrade Impact Validated**: `FastUpgradedWindowReader` provides a substantial boost in exact match accuracy over token Jaccard matching:
   - **Finance**: True E2E EM improves from **1.92% to 12.02%** (+6.26x improvement).
   - **Legal**: True E2E EM improves from **4.46% to 8.93%** (+2.0x improvement).

---

## 4. Healthcare Go/No-Go Re-Checkpoint (Task 9.4)

Per Directive 8, Healthcare ingestion was placed on hold pending clean, sanity-checked metric re-assessment.

### Threshold Audit:
- **Retrieval Benchmark**: RRF Hybrid Retrieval with `all-MiniLM-L6-v2` dense embeddings reaches **51.92% (Finance)** and **55.36% (Legal)**.
- **Reader Extraction Benchmark**: Heuristic sliding window extraction reaches **12.02% E2E EM (Finance)** and **8.93% E2E EM (Legal)**.
- **Healthcare Risk Assessment**: Healthcare clinical records (UniDoc Healthcare, 232 QA pairs) contain dense medical terminology, highly specialized clinical shorthand, and complex multi-page lab reports. Heuristic sentence-window matching without an LLM extraction reader or domain-adapted embedding model will yield low extraction accuracy (~5-10% E2E).

### Operational Directive & Recommendation:
- **Healthcare Status**: **REMAIN ON HOLD**.
- **Prerequisite for Ingestion**: Before expanding to the Healthcare tarball, retrieval accuracy must be boosted beyond 70% (e.g. via dense passage re-ranking or fine-tuned embeddings) and/or a lightweight LLM extraction prompt layer must replace keyword-window heuristics.

---

## 5. Summary of Phase 9 Completed Deliverables

1. **`tests/verify_phase9_forensic_fix.py`**: Refactored UniDoc Oracle chunk-level evaluator; verified `True_E2E <= Oracle` invariant across all test conditions.
2. **`tests/prove_determinism_back_to_back.py`**: Executed back-to-back determinism test yielding a 100% digit-for-digit match verdict (`[PASS]`).
3. **`results/phase9_forensic_audit_and_reader_revalidation.md`**: Created full Phase 9 forensic report.
4. **`results/phase5.12_unidoc_finance_legal_metrics.md`**: Appended Phase 9 dated addendum to historical ledger.
