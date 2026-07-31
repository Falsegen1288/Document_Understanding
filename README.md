# Unified Document Understanding & Layout Benchmarking Platform

> [!IMPORTANT]
> ### 📌 Featured Intern Reports
> Access all project intern reports and weekly slide decks directly:
> - 📁 **[All Intern Reports Folder](./All%20intern%20reports)**
>   - 📄 [kaliber_Week2.pdf](./All%20intern%20reports/kaliber_Week2.pdf)
>   - 📄 [Kaliber - Week 3.pdf](./All%20intern%20reports/Kaliber%20-%20Week%203.pdf)
>   - 📄 [kaliber week 4.pdf](./All%20intern%20reports/kaliber%20week%204.pdf)
>   - 📄 [Kaliber ppt week 5-6.pdf](./All%20intern%20reports/Kaliber%20ppt%20week%205-6.pdf)

A consolidated, production-ready enterprise suite for layout segmentation, OCR routing, high-fidelity table structure reconstruction, coordinate-based visual grounding, and manual ground-truth annotation. 

This repository houses three previously separate research modules in a unified, clean, and symmetric workspace:
1. **Document Ingest Pipeline & GUI** (`obsidian-precision/`)
2. **Visual Annotator Tool** (`layout_annotator/`)
3. **Consolidated Benchmarking Workspace** (`benchmarking/`)


---

## Problem Statement & Core Challenges

Traditional document ingestion pipelines suffer from the **Linear Scan Bottleneck**. By reading PDFs as a 1D stream of characters, they jumble double-column text, distort borderless tables, discard reading orders, and completely drop visual diagrams. 

A real-world document page (such as a medical catalogue, scientific journal, or invoice) is a complex, heterogeneous 2D canvas containing multiple entities simultaneously:

![Medical Catalogue Heterogeneous Canvas](docs/page_25.png)

To build a hallucination-free **Multimodal Document Q&A / RAG system**, we cannot rely on linear text. The pipeline must:
- Segment the page as a **2D spatial grid** using coordinate bounding boxes (`x0, y0, x1, y1`).
- Reconstruct borderless tables into high-fidelity cell-spanning markdown grids.
- Run coordinate-based geometry matching to ground visual figure crops with their parallel adjacent specs tables and nearby descriptions.
- Dynamic route native digital text layers and scanned image layers to distinct OCR pipelines.

---

## Multimodal Q&A Pipeline Roadmap

This repository represents **Stage 1 (Layout-Aware Ingestion & Grounding Foundation)** of our end-to-end Multimodal Document Q&A Roadmap:

```mermaid
flowchart TD
    A1["A1 · Ingestion<br/>Manifest, doc_id, content-hash,<br/>access tags (RBAC/ABAC)"]
    A2["A2 · Layout Extraction<br/>Bbox detection, NMS dedup,<br/>column-aware reading order"]
    A3["A3 · Enrichment<br/>Table markdown + image path/caption,<br/>boilerplate separated out"]
    A4["A4 · Chunking<br/>Structure-aware + contextual retrieval<br/>+ hierarchical for tables/figures"]
    A5["A5 · Embedding<br/>BGE-M3 (dense+sparse) for text/tables<br/>+ multimodal (Qwen3-VL) for images"]
    A6["A6 · Metadata Tagging<br/>Access control tags,<br/>doc type/date/department"]
    A7["A7 · Storage<br/>Vector DB (Weaviate/pgvector)<br/>+ hybrid search via RRF fusion"]
    A8["A8 · GT QA + Evaluation<br/>RAGAS (iteration) + DeepEval (CI gate)<br/>+ TruLens (monitoring)"]
    B["Phase B ·<br/>Online Query Pipeline"]

    A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7 --> A8 --> B

    classDef ingest fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef embed fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef eval fill:#FAECE7,stroke:#993C1D,color:#4A1B0C
    classDef next fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A

    class A1,A2,A3,A4 ingest
    class A5,A6,A7 embed
    class A8 eval
    class B next
```

1. **Stage 1: Document Ingest & Grounding (Active)**: Ingests PDFs, extracts bounding boxes, routes scanned regions to OCR, recovers tables, runs 2D visual grounding, and packages structured JSON.
2. **Stage 2: Multiverse Chunking, Embedding & Storage Evaluation (Future)**: An advanced multi-modal orchestration pipeline comparing various chunking strategies (Layout-Aware, Semantic, Fixed-Token) and embedding models (dense vs. sparse, late-interaction) across multiple Vector Database targets. This stage incorporates rigorous Ground Truth evaluation using RAGAS and DeepEval frameworks to evaluate retrieval accuracy, faithfulness, and context recall.
3. **Stage 3: Hybrid Graph Indexing (Future)**: Stores text blocks in vector databases while mapping layout relations (e.g. `IMAGE -[ABOVE]-> TABLE`) in Graph databases (Neo4j).
4. **Stage 4: Multimodal Agentic RAG (Future)**: Spawns an agent equipped with graph traversal and vector search tools to locate exact visual crops and generate pixel-perfect, citation-backed answers.

---

## Ingest Pipeline Workflow & Architecture

The ingestion pipeline executes sequentially across six distinct, modular stages:

![Pipeline Ingest Architecture Flowchart](docs/flowchart.png)

1. **PDF Rendering**: Converts PDF pages into high-resolution PNG frames at 150 DPI.
2. **Layout Detection**: Scans each frame to segment the canvas into specific coordinate zones (headings, paragraphs, tables, figures, footers) using `DocLayout-YOLOv10` or `NVIDIA Nemotron-Parse`.
3. **Table Grid Reconstruction**: Recovers table bounding boxes and parses complex grids (nested headers, borderless cells) into markdown table syntax via `IBM Docling TableFormer` or `TATR`.
4. **Context-Grounded VLM Captioning**: Crops figure regions and passes them to VLMs (Groq Llama-4-Scout, Local Ollama) alongside adjacent coordinate-grounded text to generate detailed, grounded captions.
5. **OCR & Digital Text Routing**: Dynamically parses native digital text zones using `PyMuPDF`/`pdfplumber`, while routing scanned/handwritten image zones to CRAFT+CRNN-based `EasyOCR` or `Tesseract`.
6. **Unified Packaging**: Compiles segmented block coordinates, OCR text, markdown tables, and captioned figures into a structured `result.json` and color-coded BBox PDF.

---

## Repository Structure

```text
Document_Understanding/
├── All intern reports/      # 📌 Featured Intern Progress Reports & Presentation Decks
│   ├── Kaliber - Week 3.pdf
│   ├── Kaliber ppt week 5-6.pdf
│   ├── kaliber week 4.pdf
│   └── kaliber_Week2.pdf
│
├── algorithms/              # Core Document_Understanding algorithms
│   ├── layout_detection/    #   Layout segmenters (YOLOv10, Nemotron-Parse, LandingAI)
│   ├── text_extraction/     #   Text extraction & OCR (PyMuPDF, EasyOCR, Tesseract, PaddleOCR)
│   ├── table_extraction/    #   Table structure recovery (IBM Docling TableFormer, TATR)
│   └── image_extraction/    #   VLM Crop Captioning (Groq Llama-4-Scout, Local Ollama)
│
├── obsidian-precision/      # Main Document Understanding Web Application
│   ├── frontend/            #   React + Vite web interface dashboard
│   └── backend/             #   FastAPI backend server & SQLite job database
│
├── layout_annotator/        # Manual BBox Ground Truth Creation Web Application
│   ├── frontend/            #   HTML5 Canvas bounding box editor interface
│   └── backend/             #   Inference backend and session save controllers
│
├── benchmarking/            # Consolidated Benchmarking & Results Workspace
│   ├── src/                 #   Benchmarking React SPA app
│   ├── results/             #   Model benchmarks, domain breakdowns & heatmaps
│   └── backend/             #   Evaluation python orchestrators & scripts
│
├── external_benchmarks/    # Standard Benchmark Datasets & Harnesses
│   ├── TAT-DQA/             #   TAT-DQA Financial Table QA Dataset
│   └── UniDoc-Bench/       #   Salesforce UniDoc Multimodal PDF Benchmark
│
├── src/                    # Core Modules (Table Indexing Strategies)
├── tests/                  # Evaluation Adapters & Test Harness Execution Scripts
├── data/                    # PDF Document datasets (Scientific, Legal, Financial, etc.)
├── notebooks/               # Analysis and visualization notebooks
├── others/                  # Auxiliary configuration, Docker, and environment setups
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── config.yaml
│   └── requirements.txt
│
├── main.py                  # CLI orchestrator entrypoint
├── download_pdfs.py        # Dataset Manager & PDF Downloader
└── README.md                # Master project overview (this file)
```

---

## Web Applications Overview

### 1. Obsidian Precision (Ingest GUI Dashboard)
A professional, dark-themed dashboard to visualize pipeline outputs interactively.
- **Features**: Interactive bounding box canvas layer with hover confidence metrics, collapsible JSON explorer, progressive SSE real-time stream, and reconstructed HTML table views with CSV exports.

![Obsidian Precision Dashboard Workspace](docs/obsidian_precision_screenshot.png)

### 2. Layout Annotator (Manual BBox GT Creator)
A specialized tool to annotate, correct, and evaluate bounding box coordinates.
- **Features**: HTML5 Canvas click-and-drag coordinate annotator, multi-layer workspace tabs (`GT` ground truth vs `DocLayoutYOLO` vs `Nemotron` vs `LandingAI`), comparative overlap metrics, and session database backups.

![Layout Annotator Workspace](docs/layout_annotator_screenshot.png)

---

## Model Evaluation Leaderboards

All models evaluated in this suite are open-source (or open-weights), allowing for local hosting, customization, and cost-effective deployment. They have been benchmarked across standard datasets (PubLayNet, DocLayNet, MedCore_Catalogue) for layout detection, text OCR, table extraction, and VLM figure analysis:

### 1. Layout Detection Benchmark Scorecard
*Evaluated across all 5 domains (Commercial, Financial, Legal, Medical, Scientific) for geometric bounding box prediction:*

| Model | Usability Rank | Reference Link | mAP@50 | mAP@50:95 | Precision | Recall | F1-Score | mean IoU | Avg Latency (s/page) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DocLayout-YOLO** | 1 | [GitHub](https://github.com/opendatalab/DocLayout-YOLO) | 0.0556 | 0.0376 | 0.1358 | 0.0638 | 0.0868 | **0.8299** | **17.27s** |
| **NVIDIA Nemotron-Parse** | 2 | [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) | **0.1110** | **0.0436** | 0.1245 | 0.0638 | 0.0844 | 0.6470 | 71.00s |

### 2. Table Extraction & Reconstruction Scorecard
*Evaluated on complex tables containing multi-row cells and borderless grids:*

| Model | Usability Rank | Reference Link | TEDS (Overall) | TEDS (Structure) | GriTS (Top) | Cell F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GLM-OCR (Official Prompt)** | 1 | [Hugging Face](https://huggingface.co/zai-org/GLM-OCR) | **0.9996** | **1.0000** | **1.0000** | **1.0000** |
| **EasyOCR + Structure** | 2 | [GitHub](https://github.com/JaidedAI/EasyOCR) | 0.9816 | 1.0000 | — | 0.8720 |
| **Table Transformer (TATR)** | 3 | [GitHub](https://github.com/microsoft/table-transformer) | 0.7444 | 0.8684 | **1.0000** | 0.4213 |
| **Docling TableFormer** | 4 | [GitHub](https://github.com/docling-project/docling) | 0.7295 | 0.7295 | 0.9828 | 0.9828 |

### 3. Text OCR (Scanned Document Engine) Scorecard
*Evaluated on non-digital image inputs to capture character error rates (CER) and word error rates (WER):*

| Model | Usability Rank | Reference Link | Corpus CER | Corpus WER | Macro CER | Macro WER | Exact Match Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EasyOCR** | 1 | [GitHub](https://github.com/JaidedAI/EasyOCR) | **10.78%** | 23.08% | **8.06%** | 19.73% | 31.25% |
| **Tesseract OCR** | 2 | [GitHub](https://github.com/tesseract-ocr/tesseract) | 13.28% | **18.32%** | 11.11% | **17.13%** | **68.75%** |
| **GLM-OCR (Page-Level)** | 3 | [Hugging Face](https://huggingface.co/zai-org/GLM-OCR) | 48.94%* | 64.41%* | — | — | — |

*\*Note: GLM-OCR CER was evaluated end-to-end on full 300 DPI page streams (absorbing line ordering & Markdown formatting), whereas EasyOCR/Tesseract were evaluated on 64 cropped element boxes.*

### 4. Vision Language Model (Figure Analysis & Captioning) Scorecard
*Evaluated on medical instrument catalogs and figures to rate description quality and attribute accuracy:*

| Model | Usability Rank | Reference Link | Type | BLEU-4 | ROUGE-L | BERTScore | Attr F1 | Avg Latency (s) | Peak VRAM |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **meta-llama/llama-4-scout-17b** | 1 | [Hugging Face](https://huggingface.co/meta-llama/Llama-Scout-17B-Instruct) | Groq API | 0.0252 | **0.2757** | **0.8329** | **0.3380** | **1.97s** | N/A (Cloud) |
| **qwen2.5vl:3b** | 2 | [GitHub](https://github.com/QwenLM/Qwen2.5-VL) | Local VLM | **0.0685** | 0.2650 | 0.8171 | 0.2722 | 41.90s | 3921 MB |
| **qwen/qwen3.6-27b** | 3 | [GitHub](https://github.com/QwenLM/Qwen) | Groq API | 0.0133 | 0.0759 | 0.7197 | 0.3019 | 5.20s | N/A (Cloud) |
| **moondream:latest** | 4 | [GitHub](https://github.com/vikhyat/moondream) | Local VLM | 0.0195 | 0.1895 | 0.5547 | 0.0278 | 14.88s | **2143 MB** |

### 5. Chunking Strategy Benchmark Scorecard
*Evaluated across 5 document chunking strategies on 37 Ground-Truth Document Questions (factual text, figure description, table lookup):*

| Strategy | Usability Rank | Recall@K | MRR (Mean Reciprocal Rank) | Factual Text Recall | Figure Description Recall | Table Lookup Recall | Downstream LLM Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`section_hierarchical`** | **1** | **0.8378** (83.78%) | 0.7085 | 0.6667 | **1.0000** (100%) | **1.0000** (100%) | **100.0%** (30/30) |
| **`element_atomic`** | **2** | 0.8108 (81.08%) | **0.7275** | **0.6667** | **1.0000** (100%) | 0.8000 (80%) | **100.0%** (29/29) |
| **`hybrid_semantic`** | **3** | 0.7838 (78.38%) | 0.6717 | 0.5556 | **1.0000** (100%) | **1.0000** (100%) | 100.0% |
| **`geometric_grounding`** | **4** | 0.7297 (72.97%) | 0.5852 | 0.5556 | 0.9286 (92.9%) | 0.8000 (80%) | — |
| **`naive_baseline`** | **5** | 0.6486 (64.86%) | 0.5492 | 0.5556 | 0.8571 (85.7%) | 0.4000 (40%) | 95.24% (20/21) |

#### Core Takeaways from Chunking Benchmarks:
1. **`section_hierarchical` Leads Overall Recall (83.78%)**: Preserving heading section paths (`section_path`) prevents cross-section context dilution and yields **100% recall on both figure descriptions and table lookups**.
2. **`element_atomic` Achieves Top MRR (0.7275)**: Treating individual layout-detected bounding box elements as atomic chunks maximizes retrieval precision and ranks target contexts higher.
3. **Naive Chunking Collapses on Tables (40% Recall)**: Arbitrary character/word splitting breaks borderless table structures, whereas structure-aware chunking doubles table retrieval accuracy.

---

### 6. Embedding Model & Retrieval Strategy Scorecard
*Evaluated across 54 ground-truth Q&A pairs using 4-stage Reciprocal Rank Fusion ($k=60$) and downstream Ragas/DeepEval faithfulness scoring (`qwen3-32b` judge):*

| Rank | Model | Strategy | Representation Type | Spec Hit Rate@5 | Overall Hit Rate@5 | Downstream Faithfulness |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | **qwen3-embedding-8b-4bit** | **bm25_hybrid** | Dense (4-bit) + Lexical | **1.00** | **0.518** | **0.454** |
| **2** | **baseline** | **bm25_only** | Pure Lexical (Regex Tokenizer) | **1.00** | **0.722** | 0.272 |
| **3** | **bge-m3** | **bm25_hybrid** | Multi-Vector + Lexical | **1.00** | 0.444 | 0.181 |
| **4** | **nv-embed-v2-fp16** | **bm25_hybrid** | Dense (FP16) + Lexical | **1.00** | 0.500 | 0.181 |
| **5** | **bge-m3** | **dense_only** | Multi-Vector (Dense Only) | **1.00** | 0.240 | 0.090 |
| **6** | **nomic-embed-text** | **bm25_hybrid** | Dense + Lexical | **1.00** | 0.462 | 0.000 |
| **7** | **bge-m3** | **3-way hybrid** | Dense + Sparse + ColBERT | 0.50 | 0.166 | 0.090 |
| **8** | **granite-vision-embedding** | **text_only** | Vision-Language (Text Only) | 0.50 | 0.363 | 0.000 |

#### Core Takeaways from Embedding & Retrieval Benchmarks:
1. **Part-Number Preservation**: Subword tokenizers fragment alphanumeric codes (e.g. `352952`). A custom regex tokenizer preserving full part-number strings hits **100% Spec Hit Rate@5**.
2. **Faithfulness Gain from Lexical Precision**: Downstream LLM faithfulness jumps from `0.090` (dense-only) to `0.454` (Qwen3-8B + BM25 hybrid) by preventing LLM spec hallucinations.
3. **RRF Hybrid Resilience**: Fusing dense semantic representations with BM25 lexical matches ($k=60$) provides superior coverage across mixed technical queries.

---

## Quick Start Guide

### 1. Environment Setup
Install dependencies inside a Python 3.10+ virtual environment:
```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Or: .venv\Scripts\activate on Windows

# Install python dependencies
pip install -r others/requirements.txt
```

Set up your model credentials in a `.env` file at the root. You can obtain the required API keys from their respective portals:
- **Groq API Key**: [Groq Console](https://console.groq.com/)
- **Gemini API Key**: [Google AI Studio](https://aistudio.google.com/)
- **LandingAI API Key**: [LandingAI Platform](https://va.landing.ai/)

```env
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
LANDING_AI_API_KEY=...
```

### 2. Run the CLI Pipeline
Run the modular ingestion pipeline directly from the root:
```bash
python main.py --pdf data/scientific/Scientific_001.pdf --layout doclayout_yolo --table docling_tableformer --figures groq
```

### 3. Launch the Obsidian Precision Website
```bash
# Start backend FastAPI server
PYTHONPATH=obsidian-precision python obsidian-precision/backend/main.py

# Start frontend Vite server
cd obsidian-precision/frontend
npm install
npm run dev
```

### 4. Launch the Manual Annotator Website
```bash
# Start annotator FastAPI server
PYTHONPATH=layout_annotator/backend python layout_annotator/backend/main.py

# Start annotator Vite server
cd layout_annotator/frontend
npm install
npm run dev
```

This will automatically spin up:
- **FastAPI Web Server** at `http://localhost:8000`
- **Redis Queue Manager** at `http://localhost:6379`
- **Celery Worker** executing pipeline extractions in the background.
To shut down the containerized system:
```bash
docker-compose down
```

---

## Conclusion & Next Steps: Guardrails, Verification & Benchmark Credibility

This Unified Document Understanding & Layout Benchmarking Platform establishes a robust 2D spatial grounding foundation. Our next phase focuses on deploying production-grade guardrails, rigorous benchmark verification, and industrial scaling:

1. **Guardrail Stack Deployment**:
   - Ship retrieval-confidence abstention gates, NLI grounding verification, numeric consistency checks, and repetition/degeneration guards.

2. **Institutionalized Cross-Benchmark Credibility**:
   - Validate performance against **TAT-DQA** and **UniDoc-Bench** with strict harness-verification discipline—auditing near-perfect scores before trusting them.

3. **Industrial-Scale Deployment, Optimization & Infrastructure**:
   - Stand up enterprise-grade serving, system latency optimization, GPU/CPU resource scaling, model quantization, and robust production deployment pipelines.



