"""
PyMuPDF Native Text Extractor
------------------------------
Extracts digitally embedded text with positional coordinates.
"""

import os
import fitz

def extract_text(pdf_path: str, pages: list[int] | None = None) -> list[dict]:
    """
    Extract native digital text with word-level bounding boxes using PyMuPDF.
    
    Args:
        pdf_path: Path to the input PDF file.
        pages: Optional list of 1-indexed page numbers. If None, extracts all.
        
    Returns:
        List of pages, each page containing a dict with:
            - page: page number
            - full_text: full joined text string
            - blocks: list of dicts with block level 'text' and 'bbox' ([x1,y1,x2,y2])
    """
    if not os.path.exists(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        return []

    try:
        doc = fitz.open(pdf_path)
        extracted_data = []

        for i, page in enumerate(doc):
            page_no = i + 1
            if pages is not None and page_no not in pages:
                continue

            # Extract structured text blocks (blocks includes bbox coordinates)
            blocks_raw = page.get_text("blocks")
            blocks = []
            full_text_parts = []

            for b in blocks_raw:
                # b format: (x0, y0, x1, y1, "text", block_no, block_type)
                x0, y0, x1, y1, text, block_no, block_type = b
                text = text.strip()
                if text:
                    blocks.append({
                        "text": text,
                        "bbox": [x0, y0, x1, y1]
                    })
                    full_text_parts.append(text)

            extracted_data.append({
                "page": page_no,
                "full_text": "\n".join(full_text_parts),
                "blocks": blocks,
                "engine": "PyMuPDF"
            })

        return extracted_data
    except Exception as e:
        print(f"[ERROR] PyMuPDF native extraction failed: {e}")
        return []
