import os
import sys
import json
import glob
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.run_real_unidoc_retrieval import extract_unidoc_evidence_sources, UniDocHybridRetriever

def main():
    print("=" * 80)
    print("      PART B: UNIDOC-BENCH REAL RETRIEVAL AUDIT (FINANCE DOMAIN)")
    print("=" * 80)

    qa_path = "external_benchmarks/UniDoc-Bench/data/QA/filtered/finance.json"
    finance_extract_dir = "external_benchmarks/UniDoc-Bench/extracted_pdfs"

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    # Gather extracted PDF files by basename
    pdf_files = glob.glob(os.path.join(finance_extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(finance_extract_dir, "**", "*.txt"), recursive=True)

    print(f"Extracted PDF/Txt Files Found: {len(pdf_files)}")

    # Map file basenames to indexed tokens (using filename + metadata string)
    doc_text_map = {}
    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        # Use filename without extension as document representation
        doc_text_map[fname] = fname.replace("_", " ").replace("-", " ").replace(".pdf", "").replace(".txt", "")

    retriever = UniDocHybridRetriever()
    if doc_text_map:
        retriever.index_documents(doc_text_map)
        print(f"Indexed {len(doc_text_map)} Finance document IDs into Hybrid Retriever.")

    text_table_qas = []
    figure_qas = []

    for item in qa_items:
        ans_type = item.get("answer_type", "")
        text_src, img_src = extract_unidoc_evidence_sources(item)
        if ans_type == "image_only" or (not text_src and img_src):
            figure_qas.append(item)
        else:
            text_table_qas.append((item, text_src))

    print(f"Total QA Pairs: {len(qa_items)} | Text/Table: {len(text_table_qas)} | Figure: {len(figure_qas)}")

    # Evaluate retrieval hit rate against evidence sources
    retrieval_hits = 0
    total_eval = len(text_table_qas)

    for item, text_src in text_table_qas:
        query = item.get("rewritten_question_obscured", "")
        top_retrieved = retriever.retrieve_top_doc(query)
        is_hit = any(os.path.basename(src) == top_retrieved or os.path.basename(src).split('.')[0] in top_retrieved for src in text_src) if text_src else False
        if is_hit:
            retrieval_hits += 1

    retrieval_acc = (retrieval_hits / total_eval) if total_eval > 0 else 0.0

    print("\n" + "=" * 80)
    print("UNIDOC-BENCH FINANCE DOMAIN RETRIEVAL RESULTS")
    print("-" * 80)
    print(f"  Indexed Corpus Documents:    {len(doc_text_map)}")
    print(f"  Text/Table QA Evaluated:     {total_eval}")
    print(f"  Figure QA (Not Supported):   {len(figure_qas)}")
    print(f"  Table/Doc Retrieval (BM25):  {retrieval_hits} / {total_eval} ({retrieval_acc:.2%})")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
