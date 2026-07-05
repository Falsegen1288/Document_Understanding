"""
Open Data Loader — PyMuPDF Layout Detector
-------------------------------------------
Uses PyMuPDF's native block extraction with font-size heuristics to classify
document elements into title, section_header, plain text, figure, figure_caption,
footnote, and page metadata — without any ML model.
"""

import os
import fitz
from PIL import Image


def load_pdf(pdf_path: str, dpi: int = 150) -> list[Image.Image]:
    """Convert all PDF pages into a list of PIL Images."""
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return []
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def get_pdf_metadata(pdf_path: str) -> dict:
    """Return dictionary of document characteristics."""
    if not os.path.exists(pdf_path):
        return {}
    doc = fitz.open(pdf_path)
    return {
        "filename": os.path.basename(pdf_path),
        "path": os.path.abspath(pdf_path),
        "page_count": len(doc),
        "file_size_kb": os.path.getsize(pdf_path) / 1024.0,
        "pages": [{"width": page.rect.width, "height": page.rect.height} for page in doc]
    }


def _classify_block(block_dict, page_width, page_height, body_font_size):
    """
    Classify a text block based on font size, position, and content.
    
    Heuristics:
      - font_size >= body * 1.4  →  title
      - font_size >= body * 1.15 →  section_header
      - font_size < body * 0.8   →  footnote
      - starts with 'Figure'     →  figure_caption
      - starts with 'Table'      →  table_caption
      - otherwise                →  plain text
    """
    lines = block_dict.get("lines", [])
    if not lines:
        return "plain text"
    
    # Get dominant font size (most common across spans)
    font_sizes = []
    all_text = []
    for line in lines:
        for span in line.get("spans", []):
            font_sizes.append(span.get("size", 0))
            all_text.append(span.get("text", ""))
    
    if not font_sizes:
        return "plain text"
    
    avg_font = sum(font_sizes) / len(font_sizes)
    text = " ".join(all_text).strip()
    
    # Position-based: top 5% or bottom 5% of page → header/footer
    bbox = block_dict["bbox"]
    y_center = (bbox[1] + bbox[3]) / 2
    block_height = bbox[3] - bbox[1]
    
    # Very small text at bottom → footnote
    if avg_font < body_font_size * 0.8:
        if y_center > page_height * 0.85:
            return "footnote"
        return "footnote"
    
    # Caption detection by content
    text_lower = text.lower().lstrip()
    if text_lower.startswith("figure") or text_lower.startswith("fig.") or text_lower.startswith("fig "):
        return "figure_caption"
    if text_lower.startswith("table"):
        return "table_caption"
    
    # Title: significantly larger font
    if avg_font >= body_font_size * 1.4:
        return "title"
    
    # Section header: moderately larger font, short text
    if avg_font >= body_font_size * 1.15 and len(text) < 200:
        return "section_header"
    
    # Sidebar/rotated text (very narrow or very tall aspect ratio)
    block_width = bbox[2] - bbox[0]
    if block_width > 0 and block_height / block_width > 5:
        return "abandon"
    
    return "plain text"


def _get_body_font_size(blocks_dict):
    """
    Determine the most common font size in the document (body text).
    """
    from collections import Counter
    size_counts = Counter()
    for b in blocks_dict:
        if b["type"] != 0:  # skip images
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                size = round(span.get("size", 0), 1)
                text = span.get("text", "").strip()
                if text:
                    size_counts[size] += len(text)
    
    if not size_counts:
        return 10.0
    return size_counts.most_common(1)[0][0]


def detect_layout(pdf_path_or_image, dpi: int = 150) -> list[dict]:
    """
    Detect document layout elements using PyMuPDF's native block extraction
    with font-size classification heuristics.
    
    Args:
        pdf_path_or_image: Path to PDF file (str) or PIL Image.
            If a PIL Image is provided, only basic block detection is available.
        dpi: Resolution for coordinate scaling (used when input is a PDF path).
    
    Returns:
        List of dicts with keys: type, bbox (in pixel coords), confidence, page.
        Types: title, section_header, plain text, figure, figure_caption,
               table_caption, footnote, abandon
    """
    # Handle PIL Image input (limited — no font info available)
    if isinstance(pdf_path_or_image, Image.Image):
        w, h = pdf_path_or_image.size
        return [{
            "type": "plain text",
            "bbox": [0.0, 0.0, float(w), float(h)],
            "confidence": 0.5,
            "page": 1
        }]
    
    pdf_path = pdf_path_or_image
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return []
    
    doc = fitz.open(pdf_path)
    all_elements = []
    
    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        pw, ph = page.rect.width, page.rect.height
        
        # Scale factor: PDF points → pixel coordinates at given DPI
        scale = dpi / 72.0
        
        # Get detailed block data with font info
        page_data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = page_data.get("blocks", [])
        
        # Determine body font size for this page
        body_font = _get_body_font_size(blocks)
        
        for b in blocks:
            bbox_pts = b["bbox"]  # in PDF points
            # Scale to pixel coordinates
            bbox_px = [
                bbox_pts[0] * scale,
                bbox_pts[1] * scale,
                bbox_pts[2] * scale,
                bbox_pts[3] * scale
            ]
            
            if b["type"] == 1:
                # Image block
                all_elements.append({
                    "type": "figure",
                    "bbox": bbox_px,
                    "confidence": 0.95,
                    "page": page_no
                })
            else:
                # Text block — classify by font size and content
                label = _classify_block(b, pw, ph, body_font)
                
                # Compute a heuristic confidence based on how clearly it matches
                conf = 0.90 if label in ("title", "figure_caption", "table_caption") else 0.85
                
                # Skip near-empty blocks
                text_content = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        text_content += span.get("text", "")
                if not text_content.strip():
                    continue
                
                all_elements.append({
                    "type": label,
                    "bbox": bbox_px,
                    "confidence": conf,
                    "page": page_no
                })
    
    # Sort by page, then top-to-bottom
    all_elements.sort(key=lambda x: (x["page"], x["bbox"][1]))
    return all_elements
