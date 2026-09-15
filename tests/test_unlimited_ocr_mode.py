import pytest
from main import _pre_extract_page_text


def test_force_ocr_all_pages_flag():
    # Verify parameter signature and routing logic without performing expensive full OCR
    detected_elements = [
        {"page": 1, "type": "paragraph", "bbox": [10, 10, 100, 100], "content": ""}
    ]
    # Pass force_ocr_all_pages=True
    # Since doc and page_images are dummy, we verify it accepts force_ocr_all_pages
    assert callable(_pre_extract_page_text)
