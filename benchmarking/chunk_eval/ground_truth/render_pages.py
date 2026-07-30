"""Render Stage 1 pages with bbox overlays for ground-truth annotation.
Requires the original PDF and PyMuPDF (fitz).
"""
import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TYPE_COLORS = {
    "title": (0.6, 0.2, 0.8),
    "plain text": (0.2, 0.4, 0.9),
    "table": (0.9, 0.5, 0.1),
    "table_caption": (0.9, 0.7, 0.3),
    "table_footnote": (0.9, 0.8, 0.5),
    "figure": (0.1, 0.7, 0.3),
    "figure_caption": (0.4, 0.8, 0.5),
    "isolate_formula": (0.8, 0.1, 0.1),
    "formula_caption": (0.9, 0.4, 0.4),
    "abandon": (0.5, 0.5, 0.5),
}

selections = [
    # Medical_004_demo_30p (Medical domain)
    ("Medical_004_demo_30p", 1, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 6, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 15, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 16, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 17, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 22, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 23, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 24, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 28, "medical/Medical_004_demo_30p.pdf"),
    ("Medical_004_demo_30p", 30, "medical/Medical_004_demo_30p.pdf"),
    
    # Researchpaper_KAI (Scientific domain)
    ("Researchpaper_KAI", 1, "scientific/Researchpaper_KAI.pdf"),
    ("Researchpaper_KAI", 2, "scientific/Researchpaper_KAI.pdf"),
    ("Researchpaper_KAI", 4, "scientific/Researchpaper_KAI.pdf"),
    ("Researchpaper_KAI", 5, "scientific/Researchpaper_KAI.pdf"),
    ("Researchpaper_KAI", 6, "scientific/Researchpaper_KAI.pdf"),
    ("Researchpaper_KAI", 8, "scientific/Researchpaper_KAI.pdf"),
    ("Researchpaper_KAI", 10, "scientific/Researchpaper_KAI.pdf"),
    
    # Scientific_001 (Scientific domain)
    ("Scientific_001", 1, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 2, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 4, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 5, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 6, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 7, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 13, "scientific/Scientific_001.pdf"),
    ("Scientific_001", 14, "scientific/Scientific_001.pdf"),
]

def render_page_with_boxes(pdf_path: Path, page_num: int, elements: list[dict], out_path: Path):
    """page_num is 1-indexed to match Stage 1's `page` field."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    page_elements = [(i, e) for i, e in enumerate(elements) if e["page"] == page_num]

    for idx, el in page_elements:
        color = TYPE_COLORS.get(el["type"], (0, 0, 0))
        rect = fitz.Rect(el["bbox"])
        page.draw_rect(rect, color=color, width=1.5)
        page.insert_text((rect.x0, max(rect.y0 - 4, 0)), f"[{idx}] {el['type']}", fontsize=6, color=color)

    pix = page.get_pixmap(dpi=150)
    pix.save(str(out_path))
    doc.close()
    logger.info(f"Rendered {out_path} ({len(page_elements)} elements)")


def render_selected_pages(selections: list[tuple[str, int, str]], data_dir: Path, output_dir: Path):
    """selections: list of (doc_stem, page_num, pdf_relative_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc_stem, page_num, pdf_rel_path in selections:
        json_path = Path("outputs") / doc_stem / f"{doc_stem}.json"
        with open(json_path, encoding="utf-8") as f:
            stage1 = json.load(f)
        pdf_path = data_dir / pdf_rel_path
        out_path = output_dir / f"{doc_stem}_p{page_num}.png"
        render_page_with_boxes(pdf_path, page_num, stage1["elements"], out_path)


if __name__ == "__main__":
    data_directory = Path("data")
    renders_directory = Path("benchmarking/chunk_eval/ground_truth/renders")
    render_selected_pages(selections, data_directory, renders_directory)
