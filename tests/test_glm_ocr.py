import pytest
from PIL import Image
from algorithms.text_extraction.glm_ocr.extractor import GLMOCRExtractor
from algorithms.table_extraction.glm_ocr.extractor import GLMOCRTableExtractor
from algorithms.image_extraction.glm_ocr.extractor import GLMOCRFigureExtractor


def test_glm_ocr_extractor_init():
    extractor = GLMOCRExtractor()
    img = Image.new("RGB", (100, 100), color="white")
    text = extractor.extract_text(img)
    assert isinstance(text, str)


def test_glm_ocr_table_extractor_init():
    extractor = GLMOCRTableExtractor()
    img = Image.new("RGB", (100, 100), color="white")
    tables = extractor.extract_table_grid(img)
    assert isinstance(tables, list)


def test_glm_ocr_figure_extractor_init():
    extractor = GLMOCRFigureExtractor()
    assert extractor is not None
    img = Image.new("RGB", (100, 100), color="white")
    res = extractor.describe_figure(img, prompt="Describe test")
    assert isinstance(res, dict)
    assert "description" in res
