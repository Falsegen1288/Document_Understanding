"""Geometric utilities for bbox proximity reasoning, used by Strategy D."""
from chunking.schema import BBox


def bbox_gap(a: BBox, b: BBox) -> float:
    """Minimum edge-to-edge Euclidean gap between two bboxes. Returns 0.0 if they
    overlap. bbox format: (x0, y0, x1, y1)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return (dx ** 2 + dy ** 2) ** 0.5


def bbox_center(box: BBox) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def is_vertically_aligned(a: BBox, b: BBox, min_overlap_ratio: float = 0.3) -> bool:
    """True if two bboxes' horizontal (x) ranges overlap by at least
    `min_overlap_ratio` of the narrower box's width — a proxy for 'same column'.
    Uses a RELATIVE ratio, not an absolute pixel tolerance, so it holds up across
    documents with different coordinate scales (points vs pixels vs normalized).
    A small negative overlap (near-touching, no true intersection) always returns
    False regardless of ratio — genuine column gutters must fail this check."""
    ax0, _, ax1, _ = a
    bx0, _, bx1, _ = b
    overlap = min(ax1, bx1) - max(ax0, bx0)
    if overlap <= 0:
        return False
    narrower_width = min(ax1 - ax0, bx1 - bx0)
    if narrower_width <= 0:
        return False
    return (overlap / narrower_width) >= min_overlap_ratio

