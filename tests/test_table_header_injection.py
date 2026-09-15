import pytest
from benchmark_harness.stages.table_header_injection import build_row_chunks, TableChunk

def test_build_row_chunks_basic():
    headers = ["Year", "Total Revenue ($M)", "Operating Income ($M)", "Diluted EPS"]
    rows = [
        ["2022", "$14,205.4", "$3,102.1", "$4.25"],
        ["2023", "$15,890.0", "$3,540.8", "$4.82"]
    ]
    chunks = build_row_chunks(
        table_id="tbl_001",
        headers=headers,
        rows=rows,
        caption="Consolidated Income Statement",
        section_title="Financial Results",
        fiscal_period_col=0
    )
    assert len(chunks) == 2
    assert chunks[1].text == (
        "Section: Financial Results | Table: Consolidated Income Statement | "
        "Row: 2023 | Total Revenue ($M) = $15,890.0 | Operating Income ($M) = "
        "$3,540.8 | Diluted EPS = $4.82"
    )
    assert chunks[0].chunk_id == "tbl_001::row_0"
    assert chunks[1].chunk_id == "tbl_001::row_1"
    assert chunks[1].metadata["table_id"] == "tbl_001"
    assert chunks[1].metadata["row_label"] == "2023"
