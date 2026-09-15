"""
Table-Header Injection for Chunking
Fixes: TAT-DQA Hit@1 floor (4.14%) caused by bare table cells ("1,204")
having no semantic anchor in the embedding space.

Drop-in target: benchmark_harness/stages/chunking.py
Applies to BOTH paths:
  (a) PDF-derived tables (Docling/TableFormer output) — baseline/glm_ocr
  (b) Pre-parsed TAT-DQA JSON tables (the skip_ocr path) — all 3 pipelines,
      until Option B rasterization lands. This fix is independent of that
      decision and should ship regardless.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TableChunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _normalize_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return str(cell).strip()


def build_row_chunks(
    table_id: str,
    headers: list[str],
    rows: list[list[Any]],
    caption: str | None = None,
    section_title: str | None = None,
    fiscal_period_col: int | None = None,
    unit_hint: str | None = None,
) -> list[TableChunk]:
    """
    Converts a structured table into one chunk PER ROW, with every cell
    anchored to its column header, the table caption, and the enclosing
    section title. This is the single highest-leverage fix identified for
    TAT-DQA retrieval: a bare cell like "1,204" is unembeddable; "Total
    Revenue ($M), FY2023: 1,204" is retrievable.

    Example output text for one row:
        "Table: Consolidated Income Statement | Section: Financial Results
         | Row: FY2023 | Total Revenue ($M) = 15,890.0 | Operating Income
         ($M) = 3,540.8 | Diluted EPS = 4.82"
    """
    chunks: list[TableChunk] = []
    header_prefix_parts = []
    if section_title:
        header_prefix_parts.append(f"Section: {section_title}")
    if caption:
        header_prefix_parts.append(f"Table: {caption}")
    header_prefix = " | ".join(header_prefix_parts)

    total_rows = len(rows)
    # Detect summary/totals row
    summary_keywords = {"total", "totals", "consolidated", "net income", "net loss", "sum", "subtotal"}
    summary_row_idx = None
    for idx, r in enumerate(rows):
        row_str = " ".join(_normalize_cell(c).lower() for c in r)
        if any(kw in row_str for kw in summary_keywords):
            summary_row_idx = idx
    if summary_row_idx is None and total_rows > 0:
        summary_row_idx = total_rows - 1

    summary_row_id = f"{table_id}::row_{summary_row_idx}" if summary_row_idx is not None else None

    for row_idx, row in enumerate(rows):
        row_label = _normalize_cell(row[fiscal_period_col]) if fiscal_period_col is not None else None
        cell_parts = []
        for col_idx, cell in enumerate(row):
            if fiscal_period_col is not None and col_idx == fiscal_period_col:
                continue
            header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
            value = _normalize_cell(cell)
            if value == "":
                continue
            cell_parts.append(f"{header} = {value}")

        text_parts = []
        if header_prefix:
            text_parts.append(header_prefix)
        if row_label:
            text_parts.append(f"Row: {row_label}")
        if unit_hint:
            text_parts.append(f"Units: {unit_hint}")
        text_parts.extend(cell_parts)
        row_text = " | ".join(text_parts)

        prev_row_id = f"{table_id}::row_{row_idx - 1}" if row_idx > 0 else None
        next_row_id = f"{table_id}::row_{row_idx + 1}" if row_idx < total_rows - 1 else None

        table_pointers = {
            "parent_table_id": table_id,
            "row_index": row_idx,
            "total_rows": total_rows,
            "prev_row_id": prev_row_id,
            "next_row_id": next_row_id,
            "summary_row_id": summary_row_id,
        }

        chunks.append(TableChunk(
            chunk_id=f"{table_id}::row_{row_idx}",
            text=row_text,
            metadata={
                "table_id": table_id,
                "row_index": row_idx,
                "row_label": row_label,
                "section_title": section_title,
                "caption": caption,
                "source": "table_row",
                "table_pointers": table_pointers,
            },
        ))
    return chunks


# ---------------------------------------------------------------------------
# TAT-DQA specific adapter.
#
# NOTE FOR ANTIGRAVITY: TAT-DQA's on-disk JSON schema needs to be confirmed
# against the actual file at ingestion time (tatdqa_dataset_dev.json). The
# public TAT-DQA / TAT-QA format typically nests a 2D `table` array (first
# row = headers) plus `paragraphs` per document. Adjust field names below
# (`doc["table"]["table"]`, etc.) to match whatever `inspect_tatdqa_schema()`
# reports before wiring this into the chunking stage.
# ---------------------------------------------------------------------------
def inspect_tatdqa_schema(sample_doc: dict) -> None:
    """Run this once against a real sample doc and print the keys/shape
    before trusting the adapter below — schemas drift between dataset
    releases."""
    print("Top-level keys:", list(sample_doc.keys()))
    if "table" in sample_doc:
        print("table keys:", list(sample_doc["table"].keys())
              if isinstance(sample_doc["table"], dict) else type(sample_doc["table"]))


def chunk_tatdqa_document(doc: dict) -> list[TableChunk]:
    """
    Adapter from a TAT-DQA document object to header-anchored row chunks.
    Assumes: doc["table"]["table"] -> list[list[str]] with row 0 = headers,
    doc["table"]["uid"] -> table id, doc.get("paragraphs") -> list of
    {"text": ...} for surrounding prose (chunked separately, unchanged).
    """
    table_obj = doc.get("table", {})
    grid: list[list[Any]] = table_obj.get("table", [])
    if not grid or len(grid) < 2:
        return []

    headers = [_normalize_cell(h) for h in grid[0]]
    data_rows = grid[1:]
    table_id = table_obj.get("uid", "unknown_table")
    caption = table_obj.get("caption")

    return build_row_chunks(
        table_id=table_id,
        headers=headers,
        rows=data_rows,
        caption=caption,
        section_title=doc.get("doc_title"),
        fiscal_period_col=0,  # TAT-DQA tables commonly use col 0 as the row label (e.g. line item or period) — VERIFY against real data
    )
