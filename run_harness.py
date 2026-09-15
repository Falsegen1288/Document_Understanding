import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import sys


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from benchmark_harness.config import PipelineConfig
from benchmark_harness.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Master Execution CLI for Phase 10 Benchmark Harness Pipelines")
    parser.add_argument("--pipeline", choices=["baseline", "glm_ocr", "unlimited_ocr"], required=True, help="Target pipeline mode")
    parser.add_argument("--dataset", choices=["tatdqa", "unidoc"], required=True, help="Target evaluation dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (for smoke testing)")
    parser.add_argument("--random-sample", action="store_true", help="Random sample queries across the dataset")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker stage")
    parser.add_argument("--use-pal", action="store_true", help="Enable PAL arithmetic executor in reading stage")
    parser.add_argument("--reader-model", type=str, default="gemini-3.6-flash", help="LLM model ID for reader (default: gemini-3.6-flash)")
    args = parser.parse_args()

    config = PipelineConfig(
        pipeline=args.pipeline,
        dataset=args.dataset,
        limit=args.limit,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
        reranking_enabled=args.rerank,
        use_pal_arithmetic=args.use_pal,
        reader_model=args.reader_model
    )

    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
