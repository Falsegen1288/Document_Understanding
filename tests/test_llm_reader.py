import pytest
from algorithms.config import GROQ_API_KEY
from src.readers.llm_reader import LLMReader


@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY is not configured")
def test_llm_reader_basic():
    reader = LLMReader()
    context = [
        "Company XYZ reported Q3 results. Total revenue for FY2023 was $42 million.",
        "Operating expenses decreased by 15 percent compared to last year."
    ]
    query = "What was the total revenue for FY2023?"
    result = reader.answer(query, context)

    assert isinstance(result, dict)
    assert result["not_found"] is False
    assert "42" in result["answer"]


@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY is not configured")
def test_llm_reader_not_found():
    reader = LLMReader()
    context = ["The weather in Paris is sunny today."]
    query = "What is the net profit of Apple Inc?"
    result = reader.answer(query, context)

    assert isinstance(result, dict)
    assert result["not_found"] is True or "NOT_FOUND" in result["answer"].upper()
