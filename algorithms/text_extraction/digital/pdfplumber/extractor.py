"""
pdfplumber Native Text Extractor
---------------------------------
Extracts digitally embedded text with positional coordinates.
"""

import os

def extract_text(pdf_path: str, pages: list[int] | None = None) -> list[dict]:
    """
    Extract native digital text with word-level bounding boxes using pdfplumber.
    
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
        import pdfplumber
        extracted_data = []

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_no = idx + 1
                if pages is not None and page_no not in pages:
                    continue

                # Extract words with their bounding boxes
                words = page.extract_words()
                blocks = []
                full_text = page.extract_text() or ""

                # Group words into rough line/block elements (or use pdfplumber's words directly)
                current_line = []
                last_top = None
                
                for w in words:
                    # w keys: 'text', 'x0', 'top', 'x1', 'bottom', 'upright', 'direction'
                    x0, y0, x1, y1 = w['x0'], w['top'], w['x1'], w['bottom']
                    text = w['text']
                    
                    # Simple heuristic to group close words into blocks
                    if last_top is None or abs(y0 - last_top) < 5:
                        current_line.append((text, [x0, y0, x1, y1]))
                    else:
                        # Save line block
                        if current_line:
                            line_text = " ".join([c[0] for c in current_line])
                            lx0 = min([c[1][0] for c in current_line])
                            ly0 = min([c[1][1] for c in current_line])
                            lx1 = max([c[1][2] for c in current_line])
                            ly1 = max([c[1][3] for c in current_line])
                            blocks.append({
                                "text": line_text,
                                "bbox": [lx0, ly0, lx1, ly1]
                            })
                        current_line = [(text, [x0, y0, x1, y1])]
                    last_top = y0

                # Append final line
                if current_line:
                    line_text = " ".join([c[0] for c in current_line])
                    lx0 = min([c[1][0] for c in current_line])
                    ly0 = min([c[1][1] for c in current_line])
                    lx1 = max([c[1][2] for c in current_line])
                    ly1 = max([c[1][3] for c in current_line])
                    blocks.append({
                        "text": line_text,
                        "bbox": [lx0, ly0, lx1, ly1]
                    })

                extracted_data.append({
                    "page": page_no,
                    "full_text": full_text,
                    "blocks": blocks,
                    "engine": "pdfplumber"
                })

        return extracted_data
    except ImportError:
        print("[ERROR] pdfplumber is not installed. Please run: pip install pdfplumber")
        return []
    except Exception as e:
        print(f"[ERROR] pdfplumber extraction failed: {e}")
        return []
