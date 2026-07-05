# Benchmark Report: Document Layout Detection on MedCore_Catalogue (1).pdf

This report evaluates **DocLayoutYOLO** (via Docling) and **NVIDIA Nemotron-Parse-v1.1** against pixel-perfect, rendering-time ground truth annotations on the updated 4-page `MedCore_Catalogue (1).pdf` dataset.

## Evaluation Methodology
- **Naive Mode**: Evaluates predictions mapped directly to canonical categories via standard mapping dicts. This captures absolute system performance including categorization errors.
- **Oracle Aligned Mode**: Matches predictions to ground truth bounding boxes spatially (IoU > 0.1) and aligns the prediction's category to the ground truth's category. This isolates purely geometric/spatial localization capabilities from classification taxonomy mismatch errors.

## 1. Overall Performance Comparison

| Metric | DocLayoutYOLO (Naive) | DocLayoutYOLO (Oracle) | Nemotron-Parse (Naive) | Nemotron-Parse (Oracle) |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@0.5 (Primary)** | **0.0409** | **0.2320** | **0.1520** | **0.3343** |
| **mAP@0.5:0.95 (Secondary)** | **0.0285** | **0.2086** | **0.0676** | **0.1763** |
| **Overall Precision (IoU=0.5)** | 0.0308 | 0.0769 | 0.1646 | 0.2195 |
| **Overall Recall (IoU=0.5)** | 0.0563 | 0.1408 | 0.1831 | 0.2535 |
| **Overall F1-Score (IoU=0.5)** | 0.0398 | 0.0995 | 0.1733 | 0.2353 |
| **Mean IoU (Matched Pairs)** | 0.7948 | 0.9012 | 0.6912 | 0.7264 |
| **Average Inference Speed** | 4.54 s/page | 4.54 s/page | 57.61 s/page | 57.61 s/page |

## 2. Per-Class Detail Scorecard (F1-Score at IoU=0.5)

| Class Name | DLY (Naive) | DLY (Oracle) | Nemotron (Naive) | Nemotron (Oracle) |
| :--- | :---: | :---: | :---: | :---: |
| **title** | 0.0000 | 0.0000 | 0.0000 | 0.1538 |
| **paragraph** | 0.0000 | 0.0000 | 0.0889 | 0.0976 |
| **table** | 0.2500 | 0.2222 | 0.8333 | 0.7692 |
| **figure** | 0.2000 | 0.2609 | 0.7500 | 0.8000 |
| **caption** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **header** | 0.0000 | 0.0000 | 0.0000 | 0.8571 |
| **footer** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **logo** | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| **list** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **section_header** | 0.0000 | 0.0690 | 0.0000 | 0.0000 |
| **product_card** | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
