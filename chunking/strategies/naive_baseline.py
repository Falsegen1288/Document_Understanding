"""Strategy A: Naive Baseline.
Flattens Stage 1 elements into raw text per page and splits with a fixed-size,
overlapping token window. Ignores element type, bbox, and structure entirely —
this is the control group we expect layout-aware strategies to beat."""
from chunking.base import BaseChunker
from chunking.schema import Chunk, union_bbox
from chunking.tokenizer_utils import count_tokens

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENC = None


class NaiveBaselineChunker(BaseChunker):
    name = "naive_baseline"

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, stage1_json: dict) -> list[Chunk]:
        doc_stem = stage1_json["metadata"]["filename"].rsplit(".", 1)[0]
        elements = stage1_json["elements"]

        # Group elements by page, preserving original list order and index.
        pages: dict[int, list[tuple[int, dict]]] = {}
        for idx, el in enumerate(elements):
            pages.setdefault(el["page"], []).append((idx, el))

        all_chunks: list[Chunk] = []
        for page in sorted(pages.keys()):
            page_elements = pages[page]
            # Flatten content with a separator, tracking which char range maps to
            # which source element index (needed so we can back-fill
            # source_element_indices per split window).
            full_text = ""
            char_to_idx: list[tuple[int, int, int]] = []  # (start, end, elem_idx)
            for idx, el in page_elements:
                start = len(full_text)
                content = el.get("content", "") or ""
                full_text += content + "\n\n"
                end = len(full_text)
                char_to_idx.append((start, end, idx))

            seq = 0
            for window_text, window_elem_indices in self._sliding_window(full_text, char_to_idx):
                if not window_text.strip():
                    continue
                bboxes = [elements[i]["bbox"] for i in window_elem_indices]
                types = [elements[i]["type"] for i in window_elem_indices]
                all_chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=stage1_json["metadata"]["filename"],
                    page=page,
                    strategy=self.name,
                    element_types=types,
                    bbox_union=union_bbox(bboxes),
                    text=window_text.strip(),
                    token_count=count_tokens(window_text),
                    source_element_indices=window_elem_indices,
                ))
                seq += 1
        return all_chunks

    def _sliding_window(self, full_text: str, char_to_idx: list[tuple[int, int, int]]):
        """Yield (window_text, contributing_element_indices) using token-based
        sliding window with overlap. NOTE: because tables/figures get flattened to
        raw content strings here (not markdown-preserved), a naive split can and
        will cut through a table mid-row — this is EXPECTED and is exactly the
        failure mode this baseline is supposed to demonstrate for the benchmark
        report. Do not special-case tables here."""
        if _ENC is not None:
            tokens = _ENC.encode(full_text)
            step = self.chunk_size - self.chunk_overlap
            for start in range(0, len(tokens), step):
                window_tokens = tokens[start:start + self.chunk_size]
                window_text = _ENC.decode(window_tokens)
                # Map back to char offsets to find contributing elements.
                char_start = len(_ENC.decode(tokens[:start]))
                char_end = char_start + len(window_text)
                contributing = [
                    idx for (s, e, idx) in char_to_idx
                    if not (e <= char_start or s >= char_end)  # overlap test
                ]
                yield window_text, contributing
                if start + self.chunk_size >= len(tokens):
                    break
        else:
            # Whitespace-approx fallback path (degraded mode, matches
            # tokenizer_utils fallback behavior).
            words = full_text.split()
            step = self.chunk_size - self.chunk_overlap
            for start in range(0, len(words), step):
                window_words = words[start:start + self.chunk_size]
                window_text = " ".join(window_words)
                contributing = [idx for (_, _, idx) in char_to_idx]  # coarse fallback
                yield window_text, contributing
                if start + self.chunk_size >= len(words):
                    break
