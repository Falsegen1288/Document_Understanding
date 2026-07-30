"""Generate pre-filled candidate boundaries/groundings for human review.
Output schema per page: benchmarking/chunk_eval/ground_truth/candidates/<doc>_p<n>.json
"""
import os
import sys
# Add project root to path for imports
sys.path.append(os.path.abspath('c:/Users/user/Downloads/Document_Understanding'))

import json
from pathlib import Path

from chunking.geometry_utils import bbox_gap
from chunking.element_types import HEADER_TYPES, TABLE_TYPES, FIGURE_TYPES

selections = [
    # Medical_004_demo_30p
    ("Medical_004_demo_30p", 1),
    ("Medical_004_demo_30p", 6),
    ("Medical_004_demo_30p", 15),
    ("Medical_004_demo_30p", 16),
    ("Medical_004_demo_30p", 17),
    ("Medical_004_demo_30p", 22),
    ("Medical_004_demo_30p", 23),
    ("Medical_004_demo_30p", 24),
    ("Medical_004_demo_30p", 28),
    ("Medical_004_demo_30p", 30),
    
    # Researchpaper_KAI
    ("Researchpaper_KAI", 1),
    ("Researchpaper_KAI", 2),
    ("Researchpaper_KAI", 4),
    ("Researchpaper_KAI", 5),
    ("Researchpaper_KAI", 6),
    ("Researchpaper_KAI", 8),
    ("Researchpaper_KAI", 10),
    
    # Scientific_001
    ("Scientific_001", 1),
    ("Scientific_001", 2),
    ("Scientific_001", 4),
    ("Scientific_001", 5),
    ("Scientific_001", 6),
    ("Scientific_001", 7),
    ("Scientific_001", 13),
    ("Scientific_001", 14),
]

def generate_candidates_for_page(doc_stem: str, page_num: int, out_dir: Path):
    with open(f"outputs/{doc_stem}/{doc_stem}.json", encoding="utf-8") as f:
        stage1 = json.load(f)
    elements = stage1["elements"]
    page_elements = [(i, e) for i, e in enumerate(elements) if e["page"] == page_num]

    boundary_candidates = [
        {"element_idx": i, "type": e["type"], "content_preview": (e.get("content") or "")[:80], "source": "title"}
        for i, e in page_elements if e["type"] in HEADER_TYPES
    ]

    # Cross-reference hybrid_semantic output for this doc/page, if it exists, to add
    # semantic_split-sourced candidates at the first element index of each sub-chunk.
    hs_path = Path(f"outputs/chunks/{doc_stem}_hybrid_semantic.json")
    if hs_path.exists():
        with open(hs_path, encoding="utf-8") as f:
            hs_chunks = json.load(f)
        for c in hs_chunks:
            if c["page"] == page_num and c["source_element_indices"]:
                first_idx = c["source_element_indices"][0]
                if not any(b["element_idx"] == first_idx for b in boundary_candidates):
                    boundary_candidates.append({
                        "element_idx": first_idx, "type": elements[first_idx]["type"],
                        "content_preview": (elements[first_idx].get("content") or "")[:80],
                        "source": "semantic_split",
                    })

    anchors = [(i, e) for i, e in page_elements if e["type"] in (TABLE_TYPES | FIGURE_TYPES)]
    text_candidates = [(i, e) for i, e in page_elements if e["type"] not in (TABLE_TYPES | FIGURE_TYPES)]
    grounding_candidates = []
    for a_idx, a_el in anchors:
        gaps = sorted(
            ((bbox_gap(a_el["bbox"], e["bbox"]), i, e) for i, e in text_candidates if i != a_idx),
            key=lambda t: t[0],
        )[:3]
        grounding_candidates.append({
            "anchor_idx": a_idx, "anchor_type": a_el["type"],
            "nearest_text": [
                {"element_idx": i, "gap": round(gap, 1), "content_preview": (e.get("content") or "")[:80]}
                for gap, i, e in gaps
            ],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_stem}_p{page_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "doc_stem": doc_stem, "page": page_num,
            "boundary_candidates": boundary_candidates,
            "grounding_candidates": grounding_candidates,
        }, f, indent=2, ensure_ascii=False)
    return out_path, len(boundary_candidates), len(grounding_candidates)


if __name__ == "__main__":
    out_dir = Path("benchmarking/chunk_eval/ground_truth/candidates")
    total_boundaries = 0
    total_groundings = 0
    for doc_stem, page_num in selections:
        _, nb, ng = generate_candidates_for_page(doc_stem, page_num, out_dir)
        total_boundaries += nb
        total_groundings += ng
    print(f"Generated candidate files for {len(selections)} pages.")
    print(f"Total boundary candidates: {total_boundaries}")
    print(f"Total grounding candidates (anchors): {total_groundings}")
