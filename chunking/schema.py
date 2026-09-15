"""Core data model for Stage 2 chunks."""
from dataclasses import dataclass, field
from typing import Literal, Optional

ElementType = Literal["title", "section_header", "text", "table", "figure"]

BBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class Chunk:
    chunk_id: str                      # format: "{doc_stem}_{strategy}_{page}_{seq:04d}"
    doc_filename: str
    page: int
    strategy: str                      # e.g. "element_atomic"
    element_types: list[ElementType]
    bbox_union: BBox                   # envelope covering all source elements
    text: str                          # final chunk text; tables inline as markdown,
                                        # figures inline as "[FIGURE CAPTION] ..."
    token_count: int
    parent_section: Optional[str] = None
    source_element_indices: list[int] = field(default_factory=list)
    page_span: list[int] = field(default_factory=list)
    lineage: dict = field(default_factory=dict)
    table_pointers: dict = field(default_factory=dict)
    cross_references: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_filename": self.doc_filename,
            "page": self.page,
            "strategy": self.strategy,
            "element_types": self.element_types,
            "bbox_union": list(self.bbox_union),
            "text": self.text,
            "token_count": self.token_count,
            "parent_section": self.parent_section,
            "source_element_indices": self.source_element_indices,
            "page_span": self.page_span if self.page_span else [self.page],
            "lineage": self.lineage,
            "table_pointers": self.table_pointers,
            "cross_references": self.cross_references,
            "metadata": self.metadata,
        }


def union_bbox(bboxes: list[BBox]) -> BBox:
    """Compute the minimal bounding box enveloping a list of bboxes.
    Raises ValueError on empty input — callers must guard against
    zero-element chunks before calling this."""
    if not bboxes:
        raise ValueError("union_bbox requires at least one bbox")
    xs0 = [b[0] for b in bboxes]
    ys0 = [b[1] for b in bboxes]
    xs1 = [b[2] for b in bboxes]
    ys1 = [b[3] for b in bboxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))
