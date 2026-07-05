# Running the Synthetic Layout Generation and Benchmarking Pipeline Locally

This guide details how to set up your laptop/local environment and execute the pipeline to generate the synthetic layout dataset, run annotations, evaluate benchmark models (DocLayoutYOLO and NVIDIA Nemotron-Parse-v1.1), and package the results.

---

## 1. Prerequisites & System Requirements

- **Operating System:** Windows, macOS, or Linux.
- **Python Version:** Python 3.10 or 3.11 is highly recommended (compatibility with PyTorch and huggingface/transformers).
- **Hardware (GPU):**
  - **CUDA GPU:** A CUDA-compatible GPU (NVIDIA) with at least **8GB VRAM** is recommended to run the NVIDIA Nemotron-Parse-v1.1 model smoothly.
  - **CPU Only:** The code will automatically fall back to CPU if no CUDA GPU is found. However, Nemotron inference will be slow (~3-5 minutes per page on CPU vs. ~30 seconds on GPU).
- **Disk Space:** ~5GB (for Hugging Face model caching of Nemotron-Parse and Docling models).

---

## 2. Environment Setup

Follow these steps from your terminal (e.g., PowerShell on Windows or Bash on macOS/Linux) inside the root project directory `medical-document-layout-annotator`:

### Step A: Initialize Virtual Environment
Create a virtual environment to isolate the pipeline dependencies:
```bash
# Create the environment named 'venv'
python -m venv venv

# Activate the environment:
# On Windows (PowerShell):
venv\Scripts\activate
# On Windows (CMD):
venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate
```

### Step B: Install Dependencies
Upgrade `pip` and install the package dependencies listed in [requirements.txt](file:///c:/Users/user/Downloads/medical-document-layout-annotator/requirements.txt):
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step C: Install Playwright Browsers
The document generators use Playwright (headless Chromium) to render HTML pages into PDFs and PNG screenshots. Install the required browser engines:
```bash
playwright install chromium
```

---

## 3. Run Pipeline Steps in Order

With your virtual environment activated, run the following steps in sequence:

### Step 1: Generate the Full Synthetic Dataset
Generates the remaining 17 samples per domain (samples 03 to 19 across all 5 domains, total 100 documents). The script automatically skips the existing 3 samples (00 to 02) to avoid overwriting them, and performs a JSON schema and boundary QA validation check on all generated documents:
```bash
python -u scratch/generate_20_per_domain.py
```
- **Generated Output:** Files are written to `output/<domain>/{json, pdfs, images}/`.
- **Expected Duration:** ~5-10 minutes (mostly HTML rendering and screenshot capturing via Playwright).

### Step 2: Run the Benchmark Evaluation
Evaluates **DocLayoutYOLO** (via Docling) and **NVIDIA Nemotron-Parse-v1.1** against your generated synthetic ground-truth annotations:
```bash
python -u scratch/run_evaluation.py
```
- **What it does:**
  - Loads DocLayoutYOLO and downloads the Nemotron model from HuggingFace (on first run, downloads ~2GB).
  - Processes each synthetic document page.
  - Aligns predicted bounding boxes with ground-truth boxes using IoU (Intersection-over-Union).
  - Computes precision, recall, F1-score, mean IoU, mAP@50, and mAP@50:95.
  - Prints evaluation summary tables to stdout.
- **Expected Duration:** ~15-30 minutes on GPU; substantially longer on CPU.

### Step 3: Compile the Jupyter Notebook
Generates the Jupyter notebook containing evaluation results, metrics, visualizations, and overlays:
```bash
python scratch/create_notebook.py
```
- **Output File:** `synthetic_evaluation_benchmark.ipynb` in the project root.
- **What it does:** Generates a structured Jupyter notebook that links to the benchmark metrics and displays visual overlays, class heatmaps, and domain breakdowns.

### Step 4: Package into a ZIP File
Compresses the dataset, metadata schemas, layout skeletons, and evaluation results into a single archive for distribution:
```bash
python scratch/package_zip.py
```
- **Output Archive:** `synthetic_layout_dataset.zip` in the project root.

---

## 4. How Skeletons are Integrated

The document layout geometries (margins, header/footer heights, columns) are derived from real layout statistics:
- **Harvest Source:** We harvested layout geometry from 500 pages of `DocLayNet-small` on Hugging Face (train split).
- **Run Notes:** Detailed statistics are documented in [layout_corpora/RUN_NOTES.md](file:///c:/Users/user/Downloads/medical-document-layout-annotator/layout_corpora/RUN_NOTES.md).
- **Re-Harvesting (Optional):** If you wish to re-run the layout geometry harvest, run:
  ```bash
  python tools/harvest_doclaynet.py --max-pages 500
  ```

---

## 5. Model Caching & Hardware Adjustments

- **Model Download Location:** Hugging Face models are cached by default in `~/.cache/huggingface/hub/`.
- **VRAM Optimizations:** To allow the pipeline to run on laptops with limited VRAM (e.g., 4GB-6GB GPUs), the Nemotron-Parse-v1.1 inference resolution has been downscaled to `1024x832` (from its native `2048x1648`) in `run_evaluation.py`, and loaded in `float16` precision:
  ```python
  nemo_processor.image_processor.final_size = (1024, 832)
  ```
