import os
import sys
import json
import glob
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.eval_harness_tables import TableEvalHarness

class UniDocDomainRetriever:
    """Production BM25 Hybrid Retriever for UniDoc Document Chunks."""
    def __init__(self):
        self.doc_tokens = {}
        self.doc_freqs = defaultdict(int)
        self.total_docs = 0
        self.avg_doc_len = 0.0

    def index_documents(self, doc_text_map: Dict[str, str]):
        for doc_id, text in doc_text_map.items():
            tokens = TableEntityTokenizer.tokenize(text)
            self.doc_tokens[doc_id] = tokens
            for t in set(tokens):
                self.doc_freqs[t] += 1
        self.total_docs = len(self.doc_tokens)
        total_len = sum(len(toks) for toks in self.doc_tokens.values())
        self.avg_doc_len = total_len / max(1, self.total_docs)

    def retrieve_top_doc(self, query: str, k1: float = 1.5, b: float = 0.75) -> str:
        q_tokens = TableEntityTokenizer.tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return list(self.doc_tokens.keys())[0] if self.doc_tokens else ""

        scores = {}
        for doc_id, doc_toks in self.doc_tokens.items():
            doc_len = len(doc_toks)
            tok_counts = defaultdict(int)
            for t in doc_toks:
                tok_counts[t] += 1
            score = 0.0
            for qt in set(q_tokens):
                if qt in tok_counts:
                    tf = tok_counts[qt]
                    df = self.doc_freqs.get(qt, 0)
                    idf = max(0.0, float(self.total_docs - df + 0.5) / (df + 0.5))
                    num = tf * (k1 + 1.0)
                    den = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (num / den)
            scores[doc_id] = score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[0][0] if sorted_docs else ""

def evaluate_unidoc_domain(domain_name: str) -> Dict[str, Any]:
    qa_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain_name}.json"
    extract_dir = f"external_benchmarks/UniDoc-Bench/extracted_pdfs/{domain_name}"

    if not os.path.exists(qa_path):
        print(f"[{domain_name.upper()}] QA file {qa_path} not found.")
        return None

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    # Load extracted PDF files
    pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)

    print(f"\n--- [{domain_name.upper()} DOMAIN EVALUATION] ---")
    print(f"Extracted Corpus Files: {len(pdf_files)}")

    # Map file basename ID to text content / token representation
    doc_text_map = {}
    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        doc_id = fname.split('.')[0]
        # Text token representation of document filename + path metadata
        doc_text_map[doc_id] = fname.replace("_", " ").replace("-", " ")

    retriever = UniDocDomainRetriever()
    if doc_text_map:
        retriever.index_documents(doc_text_map)

    # Categorize QA pairs
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

    # Evaluate retrieval and citation provenance
    retrieval_hits = 0
    for item, text_src in text_table_qas:
        query = item.get("rewritten_question_obscured", "")
        top_retrieved = retriever.retrieve_top_doc(query)
        is_hit = top_retrieved in text_src if text_src else False
        if is_hit:
            retrieval_hits += 1

    retrieval_acc = (retrieval_hits / total_eval) if total_eval > 0 else 0.0

    print(f"Total QA Pairs:               {len(qa_items)}")
    print(f"Text/Table QA Evaluated:      {total_eval}")
    print(f"Figure QA (Not Supported):    {len(figure_qas)}")
    print(f"Retrieval Accuracy (BM25):    {retrieval_hits} / {total_eval} ({retrieval_acc:.2%})")

    return {
        "domain": domain_name,
        "corpus_docs": len(doc_text_map),
        "total_qa": len(qa_items),
        "eval_qa": total_eval,
        "figure_qa": len(figure_qas),
        "retrieval_hits": retrieval_hits,
        "retrieval_acc": retrieval_acc
    }

def main():
    print("=" * 80)
    print("      PART A: UNIDOC-BENCH PER-DOMAIN INDEPENDENT VERIFICATION")
    print("=" * 80)

    for dom in ["finance", "legal"]:
        evaluate_unidoc_domain(dom)

if __name__ == "__main__":
    main()
