"""CLI entrypoint for Stage 2 chunking.
Usage (once strategies exist, from Phase 2 onward):
    python -m chunking.run_chunking --input path/to/result.json --strategy all --output outputs/chunks/
"""
import argparse
import json
import logging
from pathlib import Path

from chunking.strategies.naive_baseline import NaiveBaselineChunker
from chunking.strategies.element_atomic import ElementAtomicChunker
from chunking.strategies.section_hierarchical import SectionHierarchicalChunker
from chunking.strategies.geometric_grounding import GeometricGroundingChunker
from chunking.strategies.hybrid_semantic import HybridSemanticChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Registry populated incrementally as strategies are implemented in later phases.
STRATEGY_REGISTRY: dict[str, type] = {
    "naive_baseline": NaiveBaselineChunker,
    "element_atomic": ElementAtomicChunker,
    "section_hierarchical": SectionHierarchicalChunker,
    "geometric_grounding": GeometricGroundingChunker,
    "hybrid_semantic": HybridSemanticChunker,
}




def load_stage1_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Layout-Aware Semantic Chunking")
    parser.add_argument("--input", required=True, type=Path, help="Path to Stage 1 result.json")
    parser.add_argument(
        "--strategy",
        default="all",
        help=f"One of {list(STRATEGY_REGISTRY.keys())} or 'all'",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory for chunk JSON files")
    parser.add_argument("--validate", action="store_true", help="Run element-coverage validation after chunking")
    args = parser.parse_args()

    if not STRATEGY_REGISTRY:
        logger.error(
            "No strategies registered yet — this is expected until Phase 2 is complete. "
            "Nothing to run."
        )
        return

    args.output.mkdir(parents=True, exist_ok=True)
    stage1_json = load_stage1_json(args.input)
    doc_stem = args.input.stem

    strategies = (
        STRATEGY_REGISTRY.keys() if args.strategy == "all" else [args.strategy]
    )
    for strat_name in strategies:
        chunker_cls = STRATEGY_REGISTRY[strat_name]
        chunker = chunker_cls()
        chunks = chunker.chunk(stage1_json)
        out_path = args.output / f"{doc_stem}_{strat_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
        logger.info(f"[{strat_name}] wrote {len(chunks)} chunks -> {out_path}")

        if args.validate:
            from chunking.validation import validate_element_coverage
            allow_overlap = strat_name == "naive_baseline"
            report = validate_element_coverage(stage1_json, chunks, allow_overlap=allow_overlap)
            logger.info(f"[{strat_name}] validation: {report}")
            if not report["passed"]:
                logger.error(f"[{strat_name}] VALIDATION FAILED — dropped={report['dropped_indices']} duplicated={report['duplicated_indices']}")
                raise SystemExit(1)


if __name__ == "__main__":
    main()

