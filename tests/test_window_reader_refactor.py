import pytest
from src.readers.window_reader import FastUpgradedWindowReader


def test_fast_upgraded_window_reader_basic():
    reader = FastUpgradedWindowReader()
    dummy_context = (
        "Company XYZ reported Q3 results today. Total revenue for FY2023 was $42 million. "
        "Operating expenses decreased by 15 percent compared to last year. Net profit margin stood at 18 percent. "
        "The board expressed satisfaction with these financial outcomes."
    )
    query = "What was the total revenue for FY2023 for Company XYZ?"
    answer = reader.extract_answer_span(query, dummy_context)

    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "$42" in answer or "42 million" in answer or "$42 million" in answer or "42" in answer


def test_fast_upgraded_window_reader_empty_context():
    reader = FastUpgradedWindowReader()
    assert reader.extract_answer_span("query", "") == ""
