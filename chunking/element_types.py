"""Central registry of Stage 1 element type strings, confirmed empirically from
real outputs/*/*.json files. Discovered set (2026-07):
['abandon', 'figure', 'figure_caption', 'formula_caption', 'isolate_formula',
 'plain text', 'table', 'table_caption', 'table_footnote', 'title']
Update this file, and only this file, if a new Stage 1 output introduces a variant."""

TEXT_LIKE_TYPES = {"plain text"}
HEADER_TYPES = {"title"}  # covers BOTH document title and section headers — no
                           # separate section_header type exists in this pipeline

TABLE_TYPES = {"table"}
FIGURE_TYPES = {"figure"}
FORMULA_TYPES = {"isolate_formula"}

# Caption/footnote types ride along with their parent table/figure/formula as one
# atomic unit rather than being geometrically grounded — the type label itself is
# a more reliable pairing signal than bbox distance for these specific types.
CAPTION_TYPES = {"figure_caption", "table_caption", "table_footnote", "formula_caption"}

# Layout-detector-discarded content (running headers/footers, page numbers).
# Explicitly tracked, not silently dropped — see validate_element_coverage's
# ignored_types handling.
IGNORED_TYPES = {"abandon"}

MERGEABLE_TYPES = TEXT_LIKE_TYPES | HEADER_TYPES
ATOMIC_TYPES = TABLE_TYPES | FIGURE_TYPES | FORMULA_TYPES | CAPTION_TYPES
