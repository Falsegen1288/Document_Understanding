"""Strategy D: Geometric Proximity Grounding.
For each table/figure element, find the nearest same-page plain text elements within
a distance threshold and same-column alignment, and merge them into one grounded
chunk — regardless of their position in linear reading order. Text elements NOT
claimed by any table/figure grounding fall back to plain sequential paragraph
chunking (same floor logic as Strategy B)."""
from chunking.base import BaseChunker
from chunking.schema import Chunk, union_bbox
from chunking.tokenizer_utils import count_tokens
from chunking.element_types import TABLE_TYPES, FIGURE_TYPES, TEXT_LIKE_TYPES, MERGEABLE_TYPES, IGNORED_TYPES
from chunking.geometry_utils import bbox_gap, is_vertically_aligned

PROXIMITY_THRESHOLD = 60.0  # px gap; tune against real page coordinate scale
MAX_GROUNDED_NEIGHBORS = 3  # cap how many text blocks attach to one figure/table


class GeometricGroundingChunker(BaseChunker):
    name = "geometric_grounding"

    def __init__(self, proximity_threshold: float = PROXIMITY_THRESHOLD):
        self.proximity_threshold = proximity_threshold

    def chunk(self, stage1_json: dict) -> list[Chunk]:
        doc_stem = stage1_json["metadata"]["filename"].rsplit(".", 1)[0]
        elements = stage1_json["elements"]

        pages: dict[int, list[tuple[int, dict]]] = {}
        for idx, el in enumerate(elements):
            pages.setdefault(el["page"], []).append((idx, el))

        all_chunks: list[Chunk] = []
        for page in sorted(pages.keys()):
            page_elements = pages[page]
            
            # Filter anchors, non-ignored others, and ignored elements
            anchors = [(i, e) for i, e in page_elements if e["type"] in (TABLE_TYPES | FIGURE_TYPES)]
            others = [(i, e) for i, e in page_elements if e["type"] not in (TABLE_TYPES | FIGURE_TYPES | IGNORED_TYPES)]
            ignored = [(i, e) for i, e in page_elements if e["type"] in IGNORED_TYPES]

            claimed: set[int] = set()
            page_chunks = []
            seq = 0

            for anchor_idx, anchor_el in anchors:
                neighbors = self._find_neighbors(anchor_idx, anchor_el, others, claimed)
                group_indices = [anchor_idx] + [i for i, _ in neighbors]
                claimed.update(group_indices)

                anchor_text = (
                    anchor_el["extracted"].get("markdown", "")
                    if anchor_el["type"] in TABLE_TYPES and anchor_el.get("extracted") is not None
                    else f"[FIGURE] {anchor_el.get('content', '')}"
                )
                neighbor_texts = [e.get("content", "") or "" for _, e in neighbors]
                full_text = "\n\n".join([anchor_text] + neighbor_texts)
                bboxes = [anchor_el["bbox"]] + [e["bbox"] for _, e in neighbors]
                types = [anchor_el["type"]] + [e["type"] for _, e in neighbors]

                page_chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=stage1_json["metadata"]["filename"],
                    page=page,
                    strategy=self.name,
                    element_types=types,
                    bbox_union=union_bbox(bboxes),
                    text=full_text,
                    token_count=count_tokens(full_text),
                    source_element_indices=group_indices,
                    metadata={
                        "grounded_neighbor_count": len(neighbors),
                        "anchor_type": anchor_el["type"],
                    },
                ))
                seq += 1

            # Remaining unclaimed elements: sequential merge, same floor logic as B.
            leftover = [(i, e) for i, e in others if i not in claimed]
            for group in self._merge_leftover(leftover):
                indices = [i for i, _ in group]
                els = [e for _, e in group]
                text = self._render_leftover_text(els)
                bboxes = [e["bbox"] for e in els]
                types = [e["type"] for e in els]
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

    def _find_neighbors(self, anchor_idx, anchor_el, others, claimed):
        """Rank same-page, unclaimed plain text elements by bbox_gap to the anchor,
        filter by proximity_threshold and vertical/column alignment, return the
        closest MAX_GROUNDED_NEIGHBORS."""
        candidates = []
        for idx, el in others:
            if idx in claimed:
                continue
            if el["type"] not in TEXT_LIKE_TYPES:  # focus only on plain text elements
                continue
            gap = bbox_gap(anchor_el["bbox"], el["bbox"])
            if gap <= self.proximity_threshold and is_vertically_aligned(anchor_el["bbox"], el["bbox"]):
                candidates.append((gap, idx, el))
        candidates.sort(key=lambda t: t[0])
        return [(idx, el) for _, idx, el in candidates[:MAX_GROUNDED_NEIGHBORS]]

    def _merge_leftover(self, leftover):
        """Same small-block merge behavior as element_atomic, reused for whatever
        elements weren't claimed by any table/figure grounding."""
        groups = []
        for idx, el in leftover:
            if (
                groups
                and el["type"] in MERGEABLE_TYPES
                and groups[-1][-1][1]["type"] == el["type"]
                and count_tokens(el.get("content", "") or "") < 100
            ):
                groups[-1].append((idx, el))
            else:
                groups.append([(idx, el)])
        return groups

    def _render_leftover_text(self, els: list[dict]) -> str:
        parts = []
        for el in els:
            if el["type"] == "table" and "extracted" in el:
                parts.append(el["extracted"]["markdown"])
            elif el["type"] == "figure":
                parts.append(f"[FIGURE] {el.get('content', '')}")
            else:
                parts.append(el.get("content", "") or "")
        return "\n\n".join(p for p in parts if p)
