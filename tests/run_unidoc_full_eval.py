import os
import sys
import json
import glob
import fitz # PyMuPDF
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.eval_harness_tables import TableEvalHarness
from tests.run_real_unidoc_retrieval import extract_unidoc_evidence_sources, UniDocHybridRetriever

def build_pdf_text_index(extract_dir: str) -> Dict[str, str]:
    pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)
    
    doc_text_map = {}
    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        doc_id = fname.split('.')[0]
        try:
            if fpath.endswith(".pdf"):
                doc = fitz.open(fpath)
                text = " ".join([page.get_text() for page in doc])
                doc_text_map[doc_id] = text[:10000] # Cap text at first 10,000 chars per doc for fast search
            else:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    doc_text_map[doc_id] = fp.read()[:10000]
        except Exception:
            continue
    return doc_text_map

def evaluate_unidoc_domain_real_retrieval(domain_name: str) -> Dict[str, Any]:
    qa_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain_name}.json"
    extract_dir = f"external_benchmarks/UniDoc-Bench/extracted_pdfs/{domain_name}"

    if not os.path.exists(qa_path):
        print(f"[{domain_name.upper()}] QA JSON missing: {qa_path}")
        return None

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    print(f"\n--- [{domain_name.upper()} REAL HYBRID RETRIEVAL EVALUATION] ---")
    print(f"Extracting PDF text from {extract_dir}...")
    doc_text_map = build_pdf_text_index(extract_dir)
    print(f"Successfully indexed full text from {len(doc_text_map)} PDF documents.")

    retriever = UniDocHybridRetriever()
    if doc_text_map:
        retriever.index_documents(doc_text_map)

    text_table_qas = []
    figure_qas = []

    for item in qa_items:
        ans_type = item.get("answer_type", "")
        chunk_used = item.get("chunk_used", {})
        text_sources = []
        img_sources = []
        if isinstance(chunk_used, dict):
            for k, v in chunk_used.items():
                if isinstance(v, dict) and v.get("used"):
                    meta = str(v.get("metadata", ""))
                    if ".txt" in meta or "database" in meta:
                        fname = os.path.basename(meta)
                        doc_id = fname.split('_')[0].split('.')[0]
                        text_sources.append(doc_id)
                    elif ".jpg" in meta or ".png" in meta or "figure" in meta:
                        fname = os.path.basename(meta)
                        img_sources.append(fname)

        if ans_type == "image_only" or (not text_sources and img_sources):
            figure_qas.append(item)
        else:
            text_table_qas.append((item, text_sources))

    total_eval = len(text_table_qas)

    # Evaluate retrieval hit rate against true evidence source document IDs
    retrieval_hits = 0
    for item, text_src in text_table_qas:
        query = item.get("rewritten_question_obscured", "")
        top_retrieved = retriever.retrieve_top_doc(query)
        is_hit = top_retrieved in text_src if text_src else False
        if is_hit:
            retrieval_hits += 1

    retrieval_acc = (retrieval_hits / total_eval) if total_eval > 0 else 0.0

    print(f"  Total QA Pairs:              {len(qa_items)}")
    print(f"  Text/Table QA Evaluated:     {total_eval}")
    print(f"  Figure QA (Not Supported):   {len(figure_qas)}")
    print(f"  Table/Doc Retrieval (BM25): {retrieval_hits} / {total_eval} ({retrieval_acc:.2%})")

    return {
        "domain": domain_name,
        "docs_count": len(doc_text_map),
        "total_qa": len(qa_items),
        "eval_qa": total_eval,
        "figure_qa": len(figure_qas),
        "retrieval_hits": retrieval_hits,
        "retrieval_acc": retrieval_acc
    }

def main():
    print("=" * 80)
    print("      UNIDOC-BENCH REAL PDF TEXT HYBRID RETRIEVAL EVALUATION")
    print("=" * 80)

    for dom in ["finance", "legal"]:
        evaluate_unidoc_domain_real_retrieval(dom)

if __name__ == "__main__":
    main()
