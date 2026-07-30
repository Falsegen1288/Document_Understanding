import os
import sys
import shutil
import datetime
import argparse
import yaml
from pathlib import Path

sys.path.append("c:/Users/user/Downloads/Document_Understanding")

from embedding_bench.backends.factory import EmbeddingBackendFactory
from embedding_bench.sparse.bm25_index import BM25Index

def check_adr_status(adr_path: Path) -> bool:
    if not adr_path.exists():
        return False
    with open(adr_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for "Status\nProposed" or similar
    for line in content.split("\n"):
        if line.strip().startswith("## Status"):
            # Check next line or inline
            status_line = line.replace("## Status", "").strip()
            if not status_line:
                # Look at next line
                idx = content.find(line)
                sub = content[idx + len(line):].strip()
                status_line = sub.split("\n")[0].strip()
            
            print(f"Detected ADR Status: {status_line}")
            return "accepted" in status_line.lower()
    return False

def run_cutover(confirm: bool = False):
    if not confirm:
        raise RuntimeError("Cutover requires --confirm-cutover flag. Refusing to proceed silently.")

    adr_path = Path("docs/adr/0001-embedding-model-selection.md")
    if not check_adr_status(adr_path):
        raise RuntimeError(
            "ADR status must be flipped to 'Accepted' by a human in docs/adr/0001-embedding-model-selection.md "
            "before cutover can proceed. Refusing to run."
        )

    print("ADR status verified as 'Accepted'. Proceeding with cutover...")

    # Archive previous production embedding cache
    cache_dir = Path("outputs/.embeddings_cache")
    if cache_dir.exists():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = Path("outputs/.embeddings_cache_archive") / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        print(f"Archiving previous embeddings cache to {archive_dir}...")
        for item in cache_dir.glob("*"):
            if item.is_file():
                shutil.copy(item, archive_dir / item.name)

    # Re-embed full corpus with winning model (baseline/bm25_only)
    # BM25 requires tokenizing the full corpus and persisting the index pkl files
    print("Building and persisting production BM25 indices...")
    from embedding_bench.run_vision_sweep import load_corpus_chunks
    chunks_by_doc = load_corpus_chunks("outputs/chunks")
    
    prod_index_dir = Path("outputs/production_indices")
    prod_index_dir.mkdir(parents=True, exist_ok=True)

    for stem, chunks in chunks_by_doc.items():
        index_path = prod_index_dir / f"bm25_index_{stem}.pkl"
        print(f"Building index for {stem} -> {index_path}...")
        BM25Index.load_or_build(chunks, index_path)

    # Write new active config to experiment_config.yaml
    config_path = Path("experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config_lines = f.read()

    # Replaces the embedding_model block
    new_embedding_block = """embedding_model:
  name: "baseline"
  type: "bm25_only"
  justification: "BM25 with spec-preserving tokenizer outperformed dense/hybrid models on spec_lookup catalog matching queries."

# Runner-up fallback configuration:
# embedding_model:
#   name: "BAAI/bge-m3"
#   type: "dense_sparse"
# retrieval:
#   hybrid_fusion: "RRF (Reciprocal Rank Fusion) with k=60" """

    # Perform replace via regex or simple string replacement
    import re
    # Find block starting with embedding_model: and ending before vector_store:
    pattern = r"embedding_model:\s*\n(\s+.*\n)*"
    modified_config = re.sub(pattern, new_embedding_block + "\n\n", config_lines)

    # Update retrieval section to bm25 only
    retrieval_pattern = r"retrieval:\s*\n(\s+.*\n)*"
    new_retrieval_block = """retrieval:
  top_k: 10
  reranking: false
  hybrid_fusion: "none (bm25_only)"
  justification: "Lexical BM25 only." """
    modified_config = re.sub(retrieval_pattern, new_retrieval_block + "\n\n", modified_config)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(modified_config)

    print("Cutover completed successfully. Production configuration updated to BM25.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply winning config to production pipeline")
    parser.add_argument("--confirm-cutover", action="store_true", help="Confirm production cutover")
    args = parser.parse_args()
    
    run_cutover(confirm=args.confirm_cutover)
