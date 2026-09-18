import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from benchmark_harness.config import PipelineConfig
from benchmark_harness.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Master Execution CLI for Document Understanding Benchmark Harness")
    parser.add_argument("--pipeline", choices=["baseline", "glm_ocr", "unlimited_ocr", "custom"], default="baseline", help="Pipeline mode (default: baseline)")
    parser.add_argument("--dataset", choices=["tatdqa", "unidoc", "custom"], default="unidoc", help="Evaluation dataset (default: unidoc)")
    parser.add_argument("--dataset-file", type=str, default=None, help="Path to custom ground-truth dataset JSON file")

    # Modular custom components (active when --pipeline custom or explicitly specified)
    parser.add_argument("--layout", choices=["doclayout_yolo", "nemotron_parse", "landingai_ade"], default="doclayout_yolo", help="Layout extraction model")
    parser.add_argument("--ocr", choices=["paddleocr", "tesseract", "easyocr"], default="paddleocr", help="OCR engine")
    parser.add_argument("--table", choices=["docling_tableformer", "tatr"], default="docling_tableformer", help="Table structure extraction model")
    parser.add_argument("--figures", choices=["gemini", "groq_llama", "groq_qwen", "local_qwen", "local_moondream"], default="gemini", help="Figure transcription/captioning engine")
    parser.add_argument("--chunking", choices=["semantic_mesh", "section_hierarchical", "hybrid_semantic"], default="semantic_mesh", help="Document chunking strategy")
    parser.add_argument("--embedding", type=str, default="bge-m3", help="Embedding model ID (default: bge-m3)")
    parser.add_argument("--retrieval", choices=["rrf_hybrid", "dense", "sparse"], default="rrf_hybrid", help="Retrieval algorithm")
    parser.add_argument("--reader", "--reader-model", dest="reader_model", type=str, default="gemini-3.6-flash", help="LLM reader model (default: gemini-3.6-flash)")
    parser.add_argument("--use-symbolic-math", "--use-pal", dest="use_pal", action="store_true", help="Enable symbolic arithmetic solver")

    # Execution controls
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries evaluated (ideal for smoke testing)")
    parser.add_argument("--random-sample", action="store_true", help="Random sample queries across the dataset")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranker stage")
    args = parser.parse_args()

    # Automatically switch dataset to 'custom' if --dataset-file is provided
    dataset_name = "custom" if args.dataset_file else args.dataset

    config = PipelineConfig(
        pipeline=args.pipeline,
        dataset=dataset_name,
        dataset_file=args.dataset_file,
        layout=args.layout,
        ocr=args.ocr,
        table=args.table,
        figures=args.figures,
        chunk_strategy=args.chunking,
        embedding_model=args.embedding,
        retrieval=args.retrieval,
        reader_model=args.reader_model,
        use_pal_arithmetic=args.use_pal,
        limit=args.limit,
        random_sample=args.random_sample,
        random_seed=args.random_seed,
        reranking_enabled=args.rerank
    )

    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
