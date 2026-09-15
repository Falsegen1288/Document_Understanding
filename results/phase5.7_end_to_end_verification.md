# Diagnostic Report: Phase 5.7 End-to-End Retrieval Verification & Count Reconciliation

**Author:** Senior ML/RAG Systems Engineer  
**Project:** Document Understanding  
**Task:** Phase 5.7 — End-to-End Retrieval Verification & Count Reconciliation  
**Date:** August 2026  

---

## 1. PART A — Count Reconciliation

The discrepancy in Phase 5.6 between the total sample size (167) and the out-of-sample count (147 reported) has been fully audited and reconciled:

### Reconciled Query Dataset Counts

```
TAT-DQA Dev Set Table-Only Span Queries (Total)
├── Oracle-Present Queries (Ground truth cell present in SQLite table): 146
│   ├── Traced Sample (Phase 5.6 Detailed Analysis):              15
│   └── Remaining Oracle-Present Queries:                         131
└── Oracle-Absent Queries (Multi-cell lists, formulas, text spans):   21
                                                                  ─────
Reconciled Full Out-of-Sample Queries (167 - 15 Traced):           152
```

- **Explanation of Discrepancy**: In Phase 5.6, `test_residual_fixes.py` used a slice `table_only_span_qas[20:]` (147 queries), omitting queries 15..19. Re-evaluating out-of-sample over all **152 non-traced queries (167 − 15)** yields **Oracle-Scoped EM: 27.63% (42/152)**, matching the full sample baseline of **28.14% (47/167)**.
- **Correction Applied**: Updated `results/tatdqa_phase5.6_oracle_vs_e2e_and_residual_trace.md` with reconciled counts.

---

## 2. PART B — End-to-End Retrieval Verification Findings

### B.1 Code Citation & Retrieval Mechanism
The "End-to-End" configuration evaluated across Phase 5, 5.5, and 5.6 used **un-scoped global SQLite token search** (`src/table_indexing/strategy_a_row_kv.py`):
```python
sql = "SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count FROM entity_index e WHERE e.token IN (...) GROUP BY e.row_id, e.table_id ORDER BY hit_count DESC LIMIT 1"
cursor.execute(sql, tokens)
```
This mechanism performs a global keyword search across all 274 ingested tables simultaneously without document-level filtering or a separate BM25/dense vector index.

### B.2 Timeline & Stale Number Evidence
- **Finding**: In Phase 5.5 and Phase 5.6, when extraction fixes (multi-row header merging, section hierarchy handling, financial number cleaning) were implemented, **only the Oracle-Scoped benchmark (`table_id = doc_uid`) was re-executed**.
- The `3.59%` End-to-End figure computed in Phase 5 was **carried forward into Phase 5.5 and 5.6 reports without being re-executed against the updated extraction engine**.

### B.3 Query-Level Re-Execution Comparison
Re-executing un-scoped global search against the **current post-fix Strategy A extraction engine** produced a dramatic change in results:

- **Phase 5 Stale End-to-End Score**: 3.59% (6/167 queries)
- **Phase 5.7 Verified Re-Run Score**: **16.77% (28/167 queries)**
- **Target Table Recovery**: Out of 28 passing queries, **17 queries correctly retrieved the exact target document table (`matched_tbl_id == gt_doc_uid`) purely from un-scoped token matching**!

This proves conclusively that the `3.59%` figure was a **stale, carried-forward artifact**, not evidence of a frozen retrieval ceiling.

---

## 3. Historical Benchmark Ledger Status (Updated)

| Benchmark Metric Label | Phase 5 Score | Phase 5.5 Score | Phase 5.6 Score | Phase 5.7 Verified Baseline | Historical Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Oracle-Scoped Table-Span EM** | 10.78% | 21.56% | 28.14% | **28.14% (47/167)** | **FROZEN & APPROVED** (Official Extraction Baseline) |
| **End-to-End Table-Span EM** | 3.59% | 3.59% (stale) | 3.59% (stale) | **16.77% (28/167)** | **ACTIVE / UN-FROZEN** (Timestamped Aug 3, 2026; 3.59% **SUPERSEDED**) |
| **Target Table Presence Rate** | 12.00% | 87.43% | 87.43% | **87.43% (146/167)** | Verified Upper Bound |

---

## 4. Final Determination & Freeze Recommendation

1. **Previous `3.59%` End-to-End Figure**: Formally **SUPERSEDED & INVALIDATED** as a stale carried-forward number.
2. **Oracle-Scoped Table-Span EM (28.14%)**: **REMAINS FROZEN & APPROVED** as the official local extraction quality baseline.
3. **Fresh End-to-End Table-Span EM (16.77%)**: **UN-FROZEN**. Per guardrail instructions, this fresh number is reported and timestamped for independent review before freezing.
