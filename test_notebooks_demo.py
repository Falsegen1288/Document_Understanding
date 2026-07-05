"""
Test & Demo Script — Document Understanding Pipeline
=====================================================
Reference PDF: Henry Schein Medical (Medical_004_demo_30p.pdf) — first 30 pages
This script exercises the same code paths used by the Jupyter notebooks
and prints/saves all outputs for review.

Usage:  python test_notebooks_demo.py [--stage STAGE]
  STAGE = layout | text | table | image | all  (default: all)
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import traceback

# --- project bootstrap -------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path

from algorithms.config import (
    TEXT_REGION_LABELS,
    TABLE_REGION_LABELS,
    FIGURE_REGION_LABELS,
    DEFAULT_RENDER_DPI,
)

# --- constants ----------------------------------------------------------------
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "medical", "Medical_004_demo_30p.pdf")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "demo_test")
DPI = 150

# Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(title: str) -> None:
    """Print a fancy banner."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def sub_banner(title: str) -> None:
    """Print a sub-section banner."""
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}\n")


def load_pdf_info():
    """Load PDF and print basic info."""
    banner("PDF Information")
    doc = fitz.open(PDF_PATH)
    print(f"  PDF path  : {PDF_PATH}")
    print(f"  Exists    : {os.path.exists(PDF_PATH)}")
    print(f"  Pages     : {len(doc)}")
    print(f"  Page size : {doc[0].rect.width:.0f} x {doc[0].rect.height:.0f} pts")
    
    # Render page 1 as a sample
    pix = doc[0].get_pixmap(dpi=DPI)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    print(f"  Rendered  : {pix.width}x{pix.height} px (DPI={DPI})")
    
    return doc


def render_pages(doc, page_indices=None, dpi=DPI):
    """Render PDF pages to PIL images. Returns dict of page_num -> image."""
    if page_indices is None:
        page_indices = range(len(doc))
    
    page_images = {}
    for idx in page_indices:
        pix = doc[idx].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        page_images[idx + 1] = img
    return page_images


# =============================================================================
# STAGE 1: Layout Detection
# =============================================================================
def test_layout_detection(doc):
    banner("STAGE 1: Layout Detection — Algorithm Comparison")
    
    TEST_PAGES = [0, 1, 2]  # First 3 pages
    page_images = render_pages(doc, TEST_PAGES)
    
    # --- 1a. DocLayout-YOLO ---
    sub_banner("1a. DocLayout-YOLO (ML-based)")
    try:
        from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as yolo_detect
        
        all_elements = []
        for page_idx in TEST_PAGES:
            t0 = time.time()
            elements = yolo_detect(page_images[page_idx + 1])
            elapsed = time.time() - t0
            for el in elements:
                el["page"] = page_idx + 1
            all_elements.extend(elements)
            
            types = [e["type"] for e in elements]
            type_counts = {t: types.count(t) for t in set(types)}
            print(f"  Page {page_idx+1}: {len(elements)} elements in {elapsed:.2f}s")
            print(f"    Types: {type_counts}")
        
        print(f"\n  Total elements (YOLO, 3 pages): {len(all_elements)}")
        
        # Show sample elements
        print("\n  Sample elements from Page 1:")
        for el in all_elements:
            if el["page"] == 1:
                conf = el.get("confidence", 0)
                print(f"    {el['type']:20s} bbox={[int(x) for x in el['bbox']]}  conf={conf:.2f}")
        
        yolo_elements = all_elements
    except Exception as e:
        print(f"  [!] DocLayout-YOLO failed: {e}")
        traceback.print_exc()
        yolo_elements = []
    
    # --- 1b. open_data_loader (rule-based) ---
    sub_banner("1b. open_data_loader (Rule-Based PyMuPDF)")
    try:
        from algorithms.layout_detection.open_data_loader.extractor import detect_layout as odl_detect
        
        all_elements_odl = []
        for page_idx in TEST_PAGES:
            t0 = time.time()
            elements = odl_detect(
                pdf_path=PDF_PATH,
                page_num=page_idx,
                dpi=DPI
            )
            elapsed = time.time() - t0
            for el in elements:
                el["page"] = page_idx + 1
            all_elements_odl.extend(elements)
            
            types = [e["type"] for e in elements]
            type_counts = {t: types.count(t) for t in set(types)}
            print(f"  Page {page_idx+1}: {len(elements)} elements in {elapsed:.2f}s")
            print(f"    Types: {type_counts}")
        
        print(f"\n  Total elements (ODL, 3 pages): {len(all_elements_odl)}")
        
        # Show sample elements
        print("\n  Sample elements from Page 1:")
        for el in all_elements_odl:
            if el["page"] == 1:
                print(f"    {el['type']:20s} bbox={[int(x) for x in el['bbox']]}")
    except Exception as e:
        print(f"  [!] open_data_loader failed: {e}")
        traceback.print_exc()
    
    # --- 1c. Layout Reader ---
    sub_banner("1c. LayoutReader (LayoutLM)")
    try:
        from algorithms.layout_detection.layout_reader.extractor import detect_layout as lr_detect
        
        all_elements_lr = []
        for page_idx in TEST_PAGES:
            t0 = time.time()
            elements = lr_detect(page_images[page_idx + 1])
            elapsed = time.time() - t0
            for el in elements:
                el["page"] = page_idx + 1
            all_elements_lr.extend(elements)
            
            types = [e["type"] for e in elements]
            type_counts = {t: types.count(t) for t in set(types)}
            print(f"  Page {page_idx+1}: {len(elements)} elements in {elapsed:.2f}s")
            print(f"    Types: {type_counts}")
        
        print(f"\n  Total elements (LayoutReader, 3 pages): {len(all_elements_lr)}")
    except Exception as e:
        print(f"  [!] LayoutReader failed: {e}")
        traceback.print_exc()
    
    return yolo_elements


# =============================================================================
# STAGE 2: Text Extraction  
# =============================================================================
def test_text_extraction(doc):
    banner("STAGE 2: Text Extraction — OCR Comparison")
    
    TEST_PAGES = [0, 1, 2]
    page_images = render_pages(doc, TEST_PAGES)
    
    # First run layout detection to get text regions
    sub_banner("Running DocLayout-YOLO for text region detection...")
    try:
        from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as yolo_detect
        
        all_elements = []
        for page_idx in TEST_PAGES:
            elements = yolo_detect(page_images[page_idx + 1])
            for el in elements:
                el["page"] = page_idx + 1
            all_elements.extend(elements)
        
        # Filter text regions
        text_elements = [el for el in all_elements if el["type"] in TEXT_REGION_LABELS]
        print(f"  Text-class elements: {len(text_elements)} / {len(all_elements)} total")
        types = [e["type"] for e in text_elements]
        type_counts = {t: types.count(t) for t in set(types)}
        print(f"  Types: {type_counts}")
        
    except Exception as e:
        print(f"  [!] Layout detection failed: {e}")
        return
    
    if not text_elements:
        print("  [!] No text elements found!")
        return
    
    # Pick a sample text region
    sample = text_elements[0]
    page_img = page_images[sample["page"]]
    bbox = sample["bbox"]
    cropped = page_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    print(f"\n  Using sample text region: page={sample['page']}, type={sample['type']}")
    print(f"  Bbox: {[int(x) for x in bbox]}, Size: {cropped.size}")
    
    # --- 2a. EasyOCR ---
    sub_banner("2a. EasyOCR")
    try:
        from algorithms.text_extraction.scanned.easyocr_extractor import extract_text as easy_extract
        
        t0 = time.time()
        result = easy_extract(cropped)
        elapsed = time.time() - t0
        
        text = result if isinstance(result, str) else result.get("text", str(result))
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Text ({len(text)} chars):")
        print(f"  {text[:500]}")
    except Exception as e:
        print(f"  [!] EasyOCR failed: {e}")
        traceback.print_exc()
    
    # --- 2b. Tesseract ---
    sub_banner("2b. Tesseract OCR")
    try:
        from algorithms.text_extraction.scanned.tesseract_extractor import extract_text as tess_extract
        
        t0 = time.time()
        result = tess_extract(cropped)
        elapsed = time.time() - t0
        
        text = result if isinstance(result, str) else result.get("text", str(result))
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Text ({len(text)} chars):")
        print(f"  {text[:500]}")
    except Exception as e:
        print(f"  [!] Tesseract failed: {e}")
        traceback.print_exc()
    
    # --- 2c. PyMuPDF digital extraction ---
    sub_banner("2c. PyMuPDF (Digital Text Extraction)")
    try:
        from algorithms.text_extraction.digital.pymupdf_extractor import extract_text as pymu_extract
        
        t0 = time.time()
        result = pymu_extract(PDF_PATH, page_num=0)
        elapsed = time.time() - t0
        
        text = result if isinstance(result, str) else result.get("text", str(result))
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Text ({len(text)} chars):")
        print(f"  {text[:500]}")
    except Exception as e:
        print(f"  [!] PyMuPDF failed: {e}")
        traceback.print_exc()
    
    # --- 2d. pdfplumber digital extraction ---
    sub_banner("2d. pdfplumber (Digital Text Extraction)")
    try:
        from algorithms.text_extraction.digital.pdfplumber_extractor import extract_text as plumber_extract
        
        t0 = time.time()
        result = plumber_extract(PDF_PATH, page_num=0)
        elapsed = time.time() - t0
        
        text = result if isinstance(result, str) else result.get("text", str(result))
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Text ({len(text)} chars):")
        print(f"  {text[:500]}")
    except Exception as e:
        print(f"  [!] pdfplumber failed: {e}")
        traceback.print_exc()


# =============================================================================
# STAGE 3: Table Extraction
# =============================================================================
def test_table_extraction(doc):
    banner("STAGE 3: Table Extraction — Algorithm Comparison")
    
    # Scan ALL pages for tables
    page_images = render_pages(doc)
    
    sub_banner("Scanning all pages with DocLayout-YOLO for tables...")
    try:
        from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as yolo_detect
        
        all_elements = []
        for page_idx in range(len(doc)):
            elements = yolo_detect(page_images[page_idx + 1])
            for el in elements:
                el["page"] = page_idx + 1
            all_elements.extend(elements)
        
        print(f"  Total elements: {len(all_elements)}")
        types_all = set(e["type"] for e in all_elements)
        type_counts = {t: sum(1 for e in all_elements if e["type"] == t) for t in types_all}
        print(f"  Types: {type_counts}")
        
        # Filter table regions
        table_elements = [el for el in all_elements if el["type"] in TABLE_REGION_LABELS]
        table_pages = sorted(set(el["page"] for el in table_elements))
        
        print(f"\n  Table elements found: {len(table_elements)}")
        print(f"  Pages with tables: {table_pages}")
        
        for el in table_elements:
            conf = el.get("confidence", 0)
            print(f"    Page {el['page']}: bbox {[int(x) for x in el['bbox']]} conf={conf:.2f}")
        
    except Exception as e:
        print(f"  [!] Layout detection failed: {e}")
        traceback.print_exc()
        return
    
    if not table_elements:
        print("  [!] No tables found in the PDF!")
        return
    
    # Use first table for comparison
    sample_table = table_elements[0]
    print(f"\n  Using sample table: Page {sample_table['page']}")
    
    # --- 3a. Docling TableFormer ---
    sub_banner("3a. Docling (TableFormer)")
    try:
        from algorithms.table_extraction.docling_tableformer.extractor import extract_tables as tf_extract
        
        t0 = time.time()
        result = tf_extract(PDF_PATH, pages=[sample_table["page"]])
        elapsed = time.time() - t0
        
        print(f"  Time: {elapsed:.2f}s")
        if isinstance(result, list):
            print(f"  Tables found: {len(result)}")
            for i, tbl in enumerate(result):
                if isinstance(tbl, dict) and "markdown" in tbl:
                    md = tbl["markdown"]
                    print(f"\n  Table {i+1} (first 500 chars):")
                    print(f"  {md[:500]}")
                elif isinstance(tbl, str):
                    print(f"\n  Table {i+1} (first 500 chars):")
                    print(f"  {tbl[:500]}")
                else:
                    print(f"\n  Table {i+1}: {str(tbl)[:500]}")
        elif isinstance(result, dict):
            md = result.get("markdown", result.get("text", str(result)))
            print(f"  {md[:500]}")
        else:
            print(f"  {str(result)[:500]}")
    except Exception as e:
        print(f"  [!] Docling TableFormer failed: {e}")
        traceback.print_exc()
    
    # --- 3b. pdfplumber ---
    sub_banner("3b. pdfplumber (Rule-based)")
    try:
        from algorithms.table_extraction.pdfplumber.extractor import extract_tables as plumber_table
        
        t0 = time.time()
        result = plumber_table(PDF_PATH, pages=[sample_table["page"]])
        elapsed = time.time() - t0
        
        print(f"  Time: {elapsed:.2f}s")
        if isinstance(result, list):
            print(f"  Tables found: {len(result)}")
            for i, tbl in enumerate(result):
                if isinstance(tbl, dict) and "markdown" in tbl:
                    print(f"\n  Table {i+1} (first 500 chars):")
                    print(f"  {tbl['markdown'][:500]}")
                elif isinstance(tbl, str):
                    print(f"\n  Table {i+1} (first 500 chars):")
                    print(f"  {tbl[:500]}")
                else:
                    print(f"\n  Table {i+1}: {str(tbl)[:500]}")
        elif isinstance(result, dict):
            md = result.get("markdown", result.get("text", str(result)))
            print(f"  {md[:500]}")
        else:
            print(f"  {str(result)[:500]}")
    except Exception as e:
        print(f"  [!] pdfplumber failed: {e}")
        traceback.print_exc()
    
    # --- 3c. TATR ---
    sub_banner("3c. TATR (Table Transformer)")
    try:
        from algorithms.table_extraction.tatr.extractor import extract_tables as tatr_table
        
        page_img = page_images[sample_table["page"]]
        t0 = time.time()
        result = tatr_table(page_img)
        elapsed = time.time() - t0
        
        print(f"  Time: {elapsed:.2f}s")
        if isinstance(result, list):
            print(f"  Tables found: {len(result)}")
            for i, tbl in enumerate(result):
                print(f"\n  Table {i+1}: {str(tbl)[:500]}")
        else:
            print(f"  {str(result)[:500]}")
    except Exception as e:
        print(f"  [!] TATR failed: {e}")
        traceback.print_exc()


# =============================================================================
# STAGE 4: Image/Figure Extraction
# =============================================================================
def test_image_extraction(doc):
    banner("STAGE 4: Image/Figure Extraction — Vision API Comparison")
    
    # Scan all pages
    page_images = render_pages(doc)
    
    sub_banner("Detecting figure regions with DocLayout-YOLO...")
    try:
        from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as yolo_detect
        
        all_elements = []
        for page_idx in range(len(doc)):
            elements = yolo_detect(page_images[page_idx + 1])
            for el in elements:
                el["page"] = page_idx + 1
            all_elements.extend(elements)
        
        # Filter figure regions
        figure_elements = [el for el in all_elements if el["type"] in FIGURE_REGION_LABELS]
        
        # Crop figure regions
        figure_crops = []
        for el in figure_elements:
            page_img = page_images[el["page"]]
            bbox = el["bbox"]
            cropped = page_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            figure_crops.append({
                "image": cropped,
                "page": el["page"],
                "type": el["type"],
                "bbox": bbox,
            })
        
        if not figure_crops:
            print('  [!] No figure regions detected. Using a cropped region from page 1 as fallback.')
            img = page_images[1]
            w, h = img.size
            cropped = img.crop((int(w * 0.1), int(h * 0.3), int(w * 0.9), int(h * 0.7)))
            figure_crops.append({
                "image": cropped,
                "page": 1,
                "type": "fallback",
                "bbox": [0, 0, w, h],
            })
        
        print(f"  Figure regions found: {len(figure_crops)}")
        for c in figure_crops:
            print(f"    Page {c['page']}: {c['type']} — {c['image'].size[0]}x{c['image'].size[1]} px")
        
    except Exception as e:
        print(f"  ⚠️ Layout detection failed: {e}")
        traceback.print_exc()
        return
    
    # Only test first 2 figures to save API calls / time
    test_crops = figure_crops[:2]
    
    # --- 4a. Gemini Vision ---
    sub_banner("4a. Google Gemini 2.5 Flash")
    try:
        from algorithms.image_extraction.gemini.extractor import describe_figure as gemini_desc
        
        for i, crop in enumerate(test_crops):
            print(f"  Gemini: figure {i+1}/{len(test_crops)}...", flush=True)
            t0 = time.time()
            res = gemini_desc(crop["image"])
            elapsed = time.time() - t0
            desc = res.get("description", "")
            model = res.get("model", "N/A")
            print(f"    Model: {model}")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Description: {desc[:400] if desc else '[No description]'}")
    except Exception as e:
        print(f"  ⚠️ Gemini failed: {e}")
        traceback.print_exc()
    
    # --- 4b. Groq Vision ---
    sub_banner("4b. Groq (Llama 4 Scout)")
    try:
        from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
        
        for i, crop in enumerate(test_crops):
            print(f"  Groq: figure {i+1}/{len(test_crops)}...", flush=True)
            t0 = time.time()
            res = groq_desc(crop["image"])
            elapsed = time.time() - t0
            desc = res.get("description", "")
            model = res.get("model", "N/A")
            print(f"    Model: {model}")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Description: {desc[:400] if desc else '[No description]'}")
    except Exception as e:
        print(f"  ⚠️ Groq failed: {e}")
        traceback.print_exc()
    
    # --- 4c. GPT-4o ---
    sub_banner("4c. GPT-4o Vision")
    try:
        from algorithms.image_extraction.gpt.extractor import describe_figure as gpt_desc
        
        for i, crop in enumerate(test_crops):
            print(f"  GPT-4o: figure {i+1}/{len(test_crops)}...", flush=True)
            t0 = time.time()
            res = gpt_desc(crop["image"])
            elapsed = time.time() - t0
            desc = res.get("description", "")
            model = res.get("model", "N/A")
            print(f"    Model: {model}")
            print(f"    Time: {elapsed:.2f}s")
            print(f"    Description: {desc[:400] if desc else '[No description]'}")
    except Exception as e:
        print(f"  ⚠️ GPT-4o failed: {e}")
        traceback.print_exc()


# =============================================================================
# Main entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test Document Understanding Pipeline")
    parser.add_argument(
        "--stage",
        choices=["layout", "text", "table", "image", "all"],
        default="all",
        help="Which stage to test (default: all)",
    )
    args = parser.parse_args()
    
    banner("Document Understanding Pipeline — Demo Test")
    print(f"  Reference PDF: Henry Schein Medical Catalogue")
    print(f"  Source URL: https://www.henryschein.com/assets/Medical/8260792.pdf")
    print(f"  Demo file: Medical_004_demo_30p.pdf (first 30 pages)")
    
    doc = load_pdf_info()
    
    stage = args.stage
    
    if stage in ("layout", "all"):
        test_layout_detection(doc)
    
    if stage in ("text", "all"):
        test_text_extraction(doc)
    
    if stage in ("table", "all"):
        test_table_extraction(doc)
    
    if stage in ("image", "all"):
        test_image_extraction(doc)
    
    banner("DONE — All requested stages completed!")
    print(f"  Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
