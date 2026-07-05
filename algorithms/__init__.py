"""
algorithms — Document Understanding Pipeline Modules
=====================================================

Subpackages:
    layout_detection  : PDF structural and layout detection (DocLayout-YOLO, LayoutReader)
    text_extraction   : All text extraction from PDFs
        digital       : Native text extractors for digital PDFs (PyMuPDF, pdfplumber)
        scanned       : OCR engines for scanned PDFs (EasyOCR, Tesseract, PaddleOCR)
    table_extraction  : Table detection and structure recognition (Docling, TATR, pdfplumber)
    image_extraction  : Figure/image captioning via vision APIs (Groq, Gemini, HuggingFace, GPT)
"""
