"""Generic validation shared across all chunking strategies. Verifies that a
chunker's output correctly accounts for every Stage 1 element, being aware of ignored types."""
from chunking.schema import Chunk
from chunking.element_types import IGNORED_TYPES

def validate_element_coverage(
    stage1_json: dict,
    chunks: list[Chunk],
    allow_overlap: bool = False,
) -> dict:
    elements = stage1_json["elements"]
    total = len(elements)
    coverage_count: dict[int, int] = {i: 0 for i in range(total)}

    for c in chunks:
        for idx in c.source_element_indices:
            coverage_count[idx] = coverage_count.get(idx, 0) + 1

    dropped = [
        i for i, cnt in coverage_count.items()
        if cnt == 0 and elements[i].get("element_type") != "boilerplate"
    ]
    unexpectedly_ignored_but_covered = [
        i for i, cnt in coverage_count.items()
        if cnt == 0 and elements[i].get("element_type") == "boilerplate"
    ]
    duplicated = [i for i, cnt in coverage_count.items() if cnt > 1]


    passed = len(dropped) == 0 and (allow_overlap or len(duplicated) == 0)

    return {
        "total_elements": total,
        "covered_elements": total - len(dropped),
        "dropped_indices": dropped,
        "duplicated_indices": duplicated if not allow_overlap else [],
        "ignored_and_uncovered": unexpectedly_ignored_but_covered,  # informational only, doesn't fail validation
        "passed": passed,
    }
