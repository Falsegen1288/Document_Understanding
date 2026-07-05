"""
table_extraction package
------------------------
Extracts structured tables using Docling+TableFormer, pdfplumber, TFLOP, or TATR.
"""
from .docling_tableformer.extractor import extract_tables as docling_extract
from .tatr.extractor import extract_tables as tatr_extract

