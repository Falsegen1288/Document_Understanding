"""IBM Docling + TableFormer table extraction."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def build_converter(accurate: bool = True) -> Any:
    """Build a Docling converter with TableFormer enabled."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling is not installed. Run `uv pip install \"docling[tableformer]\"`."
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if accurate else TableFormerMode.FAST
    )
    pipeline_options.table_structure_options.do_cell_matching = True
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _page_number(item: Any) -> int | None:
    if hasattr(item, "prov") and item.prov:
        prov = item.prov[0]
        if hasattr(prov, "page_no"):
            return int(prov.page_no)
        if hasattr(prov, "page"):
            return int(prov.page)
    return None


def _bbox(item: Any) -> dict[str, float] | None:
    if not (hasattr(item, "prov") and item.prov):
        return None
    bbox = getattr(item.prov[0], "bbox", None)
    if bbox is None:
        return None
    return {
        "l": float(getattr(bbox, "l", 0.0)),
        "t": float(getattr(bbox, "t", 0.0)),
        "r": float(getattr(bbox, "r", 0.0)),
        "b": float(getattr(bbox, "b", 0.0)),
    }


def _dataframe_to_data(df: Any) -> list[list[str]]:
    if df is None:
        return []
    try:
        header = [str(col) for col in df.columns.tolist()]
        rows = df.astype(str).values.tolist()
        return [header] + rows
    except Exception:
        return []


def _fallback_dataframe(item: Any) -> Any:
    import pandas as pd

    try:
        grid = item.data.grid
        rows = [[cell.text if cell else "" for cell in row] for row in grid]
        if not rows:
            return pd.DataFrame()
        if len(rows) == 1:
            return pd.DataFrame(rows)
        return pd.DataFrame(rows[1:], columns=rows[0])
    except Exception:
        return pd.DataFrame()


def _markdown(item: Any, df: Any, doc: Any = None) -> str:
    try:
        if doc is not None:
            return item.export_to_markdown(doc=doc)
        return item.export_to_markdown()
    except Exception:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return ""


def _html(item: Any, df: Any, doc: Any = None) -> str:
    try:
        if doc is not None:
            return item.export_to_html(doc=doc)
        return item.export_to_html()
    except Exception:
        try:
            return df.to_html(index=False)
        except Exception:
            return ""


def extract_tables(
    pdf_path: str | Path,
    pages: list[int] | None = None,
    accurate: bool = True,
    converter: Any | None = None,
) -> list[dict[str, Any]]:
    """Extract tables from a PDF.

    ``pages`` uses 1-indexed page numbers to match PDF readers and notebooks.
    """
    try:
        from docling.datamodel.base_models import DocItemLabel
    except ImportError as exc:
        raise ImportError(
            "docling is not installed. Run `uv pip install \"docling[tableformer]\"`."
        ) from exc

    converter = converter or build_converter(accurate=accurate)
    result = converter.convert(str(pdf_path))
    doc = result.document
    wanted_pages = set(pages) if pages else None

    tables: list[dict[str, Any]] = []
    iterator = doc.iterate_items() if hasattr(doc, "iterate_items") else ((t, None) for t in doc.tables)
    for item, _level in iterator:
        if hasattr(item, "label") and item.label != DocItemLabel.TABLE:
            continue

        page = _page_number(item)
        if wanted_pages is not None and page not in wanted_pages:
            continue

        try:
            df = item.export_to_dataframe(doc=doc)
        except Exception:
            try:
                df = item.export_to_dataframe()
            except Exception:
                df = _fallback_dataframe(item)

        tables.append(
            {
                "table_index": len(tables) + 1,
                "page": page,
                "bbox": _bbox(item),
                "data": _dataframe_to_data(df),
                "dataframe": df,
                "markdown": _markdown(item, df, doc),
                "html": _html(item, df, doc),
                "engine": "Docling+TableFormer",
            }
        )

    return tables


def extract_tables_from_image(image: Image.Image, accurate: bool = True) -> list[dict[str, Any]]:
    """Best-effort image table extraction by saving a temporary image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp_path = Path(temp.name)
        image.convert("RGB").save(temp_path)
    try:
        return extract_tables(temp_path, accurate=accurate)
    finally:
        temp_path.unlink(missing_ok=True)


run_docling_tableformer = extract_tables
