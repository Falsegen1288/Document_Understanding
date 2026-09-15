# Phase 5.11 Report: UniDoc-Bench Full Remediation (Production Ingestion + Genuine Hybrid Retrieval)

**Author:** Senior ML/RAG Systems Engineer  
**Project:** Document Understanding  
**Task:** Phase 5.11 — UniDoc Production Ingestion Stack, Genuine RRF Hybrid Retrieval & Re-Verification  
**Date:** August 2026  

---

## 1. PART A — Healthcare Download & Extraction Status

- **Archive Size on Disk**: `external_benchmarks/UniDoc-Bench/pdf_archives/healthcare_pdfs.tar.gz` = **594.00 MB** (out of ~1.28 GB expected).
- **Extracted Corpus**: **133 Healthcare PDF Documents** extracted cleanly into `external_benchmarks/UniDoc-Bench/extracted_pdfs/healthcare/`.
- **Status Determination**: Download remains active in background. Per Phase 5.11 instructions, Healthcare work is scoped as **NOT YET (Download Active)** without holding back Finance and Legal.

---

## 2. PART B & C — Genuine Production Ingestion & RRF Hybrid Retrieval

### B.1 Production Ingestion Pipeline Evidence
PDF document corpora were ingested into the production Strategy A SQLite store using layout segmentation, column header normalization, and cell-value token indexing:

- **UniDoc-Bench Finance**: **621 PDF Documents** (18,630 table/prose rows populated into Strategy A SQLite tables).
- **UniDoc-Bench Legal**: **909 PDF Documents** (27,208 table/prose rows populated into Strategy A SQLite tables).
- **Code Citation**: Ingestion executed via `StrategyARowKVIndex.ingest_tables()` in `src/table_indexing/strategy_a_row_kv.py` and `tests/run_unidoc_remediation_pipeline.py`.

### C.1 Genuine RRF Hybrid Retrieval Evidence
Implemented `ProductionRRFHybridRetriever` fusing dense embedding vector similarity with sparse BM25 token matching using Reciprocal Rank Fusion (RRF, $k=60$):

$$RRF(d) = \frac{1}{60 + \text{rank}_{bm25}(d)} + \frac{1}{60 + \text{rank}_{dense}(d)}$$

- **Code Citation**: `ProductionRRFHybridRetriever.retrieve_top_document_rrf()` in `tests/run_unidoc_remediation_pipeline.py`.

---

## 3. PART D & E — Final Operational Status Table

| Benchmark Dataset / Domain | QA Pair Count | Ingested Document Count | Genuine Hybrid Retrieval (Dense+BM25) | Production Ingestion Stack | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TAT-DQA (Dev Set)** | 1,644 | 274 Tables | **PASS** | **PASS** | **PASS** |
| **UniDoc-Bench (Finance)** | 208 | 621 PDFs | **PASS** *(RRF Dense+BM25)* | **PASS** *(Strategy A SQLite, 18.6k rows)* | **NOT YET** *(0.0% E2E EM, 6.25% Retrieval)* |
| **UniDoc-Bench (Legal)** | 224 | 909 PDFs | **PASS** *(RRF Dense+BM25)* | **PASS** *(Strategy A SQLite, 27.2k rows)* | **NOT YET** *(0.0% E2E EM, 5.80% Retrieval)* |
| **UniDoc-Bench (Healthcare)** | 232 | 133 PDFs | Pending Tarball | Pending Tarball | **NOT YET** *(594 MB downloaded)* |

---

## 4. Trigger Condition for Institutionalization

**TAT-DQA is the single benchmark dataset that has earned an explicit PASS** across all six operational criteria (genuine hybrid retrieval, production-pipeline ingestion, citation-aware scoring, oracle/E2E separation, non-stale metrics, disk file integrity). UniDoc-Bench domains remain marked **NOT YET** due to domain transfer gaps on unstructured PDF prose.
