import re
from typing import List, Dict, Any, Optional
from .tokenizer import TableEntityTokenizer

class FlattenedTableBaselineIndex:
    """
    Baseline table indexing implementation:
    Linearizes structured tables into flattened markdown/HTML text blocks,
    chunks them, and executes RRF BM25 + string matching.
    """

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []

    def ingest_tables(self, tables: List[Dict[str, Any]]) -> None:
        """
        Flatten table grid into text chunk with metadata:
        section_path, element_type='table', page, bbox
        """
        self.chunks = []
        for table in tables:
            table_id = table.get("table_id")
            section_path = table.get("section_path", "")
            page = table.get("page", 1)
            bbox = table.get("bbox", [0, 0, 0, 0])
            headers = table.get("column_headers", [])
            rows = table.get("rows", [])

            # Construct flattened markdown text
            md_lines = []
            if headers:
                header_line = "| " + " | ".join(headers) + " |"
                sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                md_lines.append(header_line)
                md_lines.append(sep_line)

            for r in rows:
                row_label = r.get("row_label", "")
                cells = r.get("cell_values", [])
                row_str = "| " + " | ".join(cells) + " |"
                md_lines.append(row_str)

            flattened_text = "\n".join(md_lines)

            # Store chunk
            chunk = {
                "chunk_id": f"chunk_{table_id}",
                "table_id": table_id,
                "element_type": "table",
                "section_path": section_path,
                "page": page,
                "bbox": bbox,
                "text": flattened_text,
                "tokens": TableEntityTokenizer.tokenize(flattened_text),
                "headers": headers,
                "rows": rows
            }
            self.chunks.append(chunk)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Execute BM25/keyword search over flattened table text chunks.
        Returns top-k chunks with provenance citations.
        """
        query_tokens = TableEntityTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scored_chunks = []
        for chunk in self.chunks:
            # Score based on token overlap / term frequency
            score = 0
            chunk_tokens = chunk["tokens"]
            for q_tok in query_tokens:
                score += chunk_tokens.count(q_tok)

            if score > 0:
                # Extract best candidate cell value from raw text (imperfect baseline extraction)
                extracted_cell_value = self._extract_cell_from_flattened(chunk, query)
                
                result = {
                    "score": score,
                    "table_id": chunk["table_id"],
                    "section_path": chunk["section_path"],
                    "page": chunk["page"],
                    "bbox": chunk["bbox"],
                    "extracted_value": extracted_cell_value,
                    "retrieved_context": chunk["text"],
                    "citation": {
                        "row_label": None,  # Baseline cannot isolate exact row label reliably
                        "column_label": None,  # Baseline cannot isolate exact column label reliably
                        "table_id": chunk["table_id"],
                        "section_path": chunk["section_path"],
                        "page": chunk["page"],
                        "bbox": chunk["bbox"]
                    }
                }
                scored_chunks.append(result)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]

    def _extract_cell_from_flattened(self, chunk: Dict[str, Any], query: str) -> Optional[str]:
        """Naive cell extraction from flattened text blob."""
        q_tokens = set(TableEntityTokenizer.tokenize(query))
        best_row = None
        max_overlap = 0

        for r in chunk["rows"]:
            row_text = " ".join([r.get("row_label", "")] + r.get("cell_values", []))
            r_tokens = set(TableEntityTokenizer.tokenize(row_text))
            overlap = len(q_tokens.intersection(r_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
                best_row = r

        if best_row:
            # Return last cell or matching cell
            return best_row.get("cell_values", [""])[-1] if best_row.get("cell_values") else None
        return None
