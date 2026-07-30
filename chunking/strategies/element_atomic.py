"""Strategy B: Element-Atomic Chunking.
Each Stage 1 element becomes its own chunk. Small text blocks
below a token floor get merged with the next element of the SAME type (to avoid
tiny useless chunks). Tables and figures are NEVER merged or split, regardless of
size — their structural fidelity (exact markdown, exact caption) is preserved
byte-for-byte."""
from chunking.base import BaseChunker
from chunking.schema import Chunk, union_bbox
from chunking.tokenizer_utils import count_tokens
from chunking.element_types import MERGEABLE_TYPES, IGNORED_TYPES

MERGE_TOKEN_FLOOR = 100


class ElementAtomicChunker(BaseChunker):
    name = "element_atomic"

    def __init__(self, merge_token_floor: int = MERGE_TOKEN_FLOOR):
        self.merge_token_floor = merge_token_floor

    def chunk(self, stage1_json: dict) -> list[Chunk]:
        doc_stem = stage1_json["metadata"]["filename"].rsplit(".", 1)[0]
        elements = stage1_json["elements"]

        pages: dict[int, list[tuple[int, dict]]] = {}
        for idx, el in enumerate(elements):
            pages.setdefault(el["page"], []).append((idx, el))

        all_chunks: list[Chunk] = []
        for page in sorted(pages.keys()):
            page_elements = pages[page]
            
            # Filter non-ignored and ignored elements
            non_ignored = [(idx, el) for idx, el in page_elements if el["type"] not in IGNORED_TYPES]
            ignored = [(idx, el) for idx, el in page_elements if el["type"] in IGNORED_TYPES]
            
            groups = self._merge_small_blocks(non_ignored)

            page_chunks = []
            seq = 0
            for group in groups:
                indices = [idx for idx, _ in group]
                els = [el for _, el in group]
                text = self._render_group_text(els)
                bboxes = [el["bbox"] for el in els]
                types = [el["type"] for el in els]
                metadata = {}
                # Preserve table/figure-specific fields for downstream fidelity checks.
                for el in els:
                    if el["type"] == "table" and el.get("extracted") is not None:
                        metadata["table_markdown"] = el["extracted"].get("markdown", "")
                        metadata["table_extractor"] = el["extracted"].get("extractor")
                    if el["type"] == "figure":
                        metadata["figure_caption"] = el.get("content")

                page_chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=stage1_json["metadata"]["filename"],
                    page=page,
                    strategy=self.name,
                    element_types=types,
                    bbox_union=union_bbox(bboxes),
                    text=text,
                    token_count=count_tokens(text),
                    source_element_indices=indices,
                    metadata=metadata,
                ))
                seq += 1

            # Attach ignored elements to the nearest chunk on this page
            for ignored_idx, ignored_el in ignored:
                if page_chunks:
                    from chunking.geometry_utils import bbox_gap
                    best_chunk = min(
                        page_chunks,
                        key=lambda c: bbox_gap(ignored_el["bbox"], c.bbox_union)
                    )
                    best_chunk.source_element_indices.append(ignored_idx)

            all_chunks.extend(page_chunks)
        return all_chunks

    def _merge_small_blocks(self, page_elements):
        """Walk elements in order; merge consecutive small blocks of the same
        mergeable type. Atomic elements (tables/figures/captions) always start
        a new group and are never merged."""
        groups: list[list[tuple[int, dict]]] = []
        for idx, el in page_elements:
            el_type = el["type"]
            if el_type not in MERGEABLE_TYPES:
                groups.append([(idx, el)])
                continue

            content_tokens = count_tokens(el.get("content", "") or "")
            can_merge_into_prev = (
                groups
                and content_tokens < self.merge_token_floor
                and all(t in MERGEABLE_TYPES for _, e in groups[-1] for t in [e["type"]])
                and groups[-1][-1][1]["type"] == el_type
            )
            if can_merge_into_prev:
                groups[-1].append((idx, el))
            else:
                groups.append([(idx, el)])
        return groups

    def _render_group_text(self, els: list[dict]) -> str:
        """Render a group of elements into final chunk text. Tables render as their
        preserved markdown, figures render with an explicit tag."""
        parts = []
        for el in els:
            if el["type"] == "table" and el.get("extracted") is not None:
                parts.append(el["extracted"].get("markdown", ""))
            elif el["type"] == "figure":
                parts.append(f"[FIGURE] {el.get('content', '')}")
            else:
                parts.append(el.get("content", "") or "")
        return "\n\n".join(p for p in parts if p)
