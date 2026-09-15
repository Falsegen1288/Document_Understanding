import os
import sys
import json
import glob
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.eval_harness_tables import TableEvalHarness

def extract_unidoc_evidence_sources(item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    text_sources = []
    image_sources = []
    chunk_used = item.get("chunk_used", {})
    if isinstance(chunk_used, dict):
        for k, v in chunk_used.items():
            if isinstance(v, dict) and v.get("used", False):
                meta = v.get("metadata")
                if isinstance(meta, dict) and "source" in meta:
                    text_sources.append(os.path.basename(meta["source"]))
                elif isinstance(meta, str):
                    if meta.endswith((".jpg", ".png", ".jpeg")):
                        image_sources.append(os.path.basename(meta))
                    else:
                        text_sources.append(os.path.basename(meta))
    return text_sources, image_sources

class UniDocHybridRetriever:
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

def audit_domain(domain_name: str):
    qa_path = os.path.join("external_benchmarks", "UniDoc-Bench", "data", "QA", "filtered", f"{domain_name}.json")
    extract_dir = os.path.join("external_benchmarks", "UniDoc-Bench", "extracted_pdfs", domain_name)

    if not os.path.exists(qa_path):
        print(f"[{domain_name.upper()}] QA JSON missing.")
        return None

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    # Build document text store from extracted PDFs
    doc_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)

    doc_text_map = {}
    for fpath in doc_files:
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                doc_text_map[fname] = fp.read()
        except Exception:
            continue

    retriever = UniDocHybridRetriever()
    if doc_text_map:
        retriever.index_documents(doc_text_map)

    text_table_qas = []
    figure_qas = []

    for item in qa_items:
        ans_type = item.get("answer_type", "")
        text_src, img_src = extract_unidoc_evidence_sources(item)
        if ans_type == "image_only" or (not text_src and img_src):
            figure_qas.append(item)
        else:
            text_table_qas.append((item, text_src))

    # Evaluate retrieval and provenance accuracy
    retrieval_hits = 0
    total_eval = len(text_table_qas)

    for item, text_src in text_table_qas:
        query = item.get("rewritten_question_obscured", "")
        top_retrieved = retriever.retrieve_top_doc(query)
        
        # Check if retrieved doc matches any ground truth evidence source file
        is_hit = any(os.path.basename(src) == top_retrieved for src in text_src) if text_src else False
        if is_hit:
            retrieval_hits += 1

    retrieval_acc = (retrieval_hits / total_eval) if total_eval > 0 else 0.0

    print(f"\n--- [{domain_name.upper()} DOMAIN RESULTS] ---")
    print(f"  Extracted Document Files:   {len(doc_text_map)}")
    print(f"  Total QA Pairs:             {len(qa_items)}")
    print(f"  Text/Table QA Evaluated:    {total_eval}")
    print(f"  Figure QA (Not Supported):  {len(figure_qas)}")
    print(f"  Retrieval Accuracy (BM25):  {retrieval_hits} / {total_eval} ({retrieval_acc:.2%})")

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
    print("      PART B: UNIDOC-BENCH REAL HYBRID RETRIEVAL & CITATION AUDIT")
    print("=" * 80)

    domains = ["finance", "legal", "healthcare"]
    results = {}
    for dom in domains:
        res = audit_domain(dom)
        if res:
            results[dom] = res

if __name__ == "__main__":
    main()
