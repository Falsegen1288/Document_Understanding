"""Central configuration for the document understanding project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

CATEGORIES = ["commercial", "financial", "legal", "medical", "scientific"]

DEFAULT_DOCLAYOUT_MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
DEFAULT_DOCLAYOUT_MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DEFAULT_DOCLAYOUT_IMAGE_SIZE = 1024
DEFAULT_DOCLAYOUT_CONFIDENCE = 0.20
DEFAULT_RENDER_DPI = 200

DOCLAYOUT_CLASS_NAMES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

TEXT_REGION_LABELS = {
    "title",
    "plain text",
    "text",
    "paragraph",
    "figure_caption",
    "table_caption",
    "table_footnote",
    "formula_caption",
    "caption",
}
TABLE_REGION_LABELS = {"table"}
FIGURE_REGION_LABELS = {"figure", "image", "picture"}

load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or GEMINI_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANDING_AI_API_KEY = os.getenv("LANDING_AI_API_KEY")

QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY") or os.getenv("BAIDU_API_KEY")
QIANFAN_SECRET_KEY = os.getenv("QIANFAN_SECRET_KEY") or os.getenv("BAIDU_SECRET_KEY")
BAIDU_API_KEY = QIANFAN_API_KEY
BAIDU_SECRET_KEY = QIANFAN_SECRET_KEY



def ensure_project_dirs() -> None:
    """Create project folders that are expected at runtime."""
    for path in (DATA_DIR, OUTPUT_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(category: str, doc_id: str) -> Path:
    """Return outputs/<category>/<doc_id>, creating it if needed."""
    output_dir = OUTPUT_DIR / category / doc_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_csv_path(category: str) -> Path:
    """Return the CSV manifest path for a known category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Expected one of: {CATEGORIES}")
    csv_path = DATA_DIR / category / f"{category}_pdfs.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path
