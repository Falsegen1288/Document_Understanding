import os
import sys
import json
import glob
import re
import fitz # PyMuPDF
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.run_unidoc_dense_pipeline import SentenceTransformerRRFRetriever, check_answer_match
from tests.run_unidoc_full_metrics import parse_unidoc_gt_doc_ids

from src.readers.window_reader import FastUpgradedWindowReader  # noqa: F401 (moved to src/readers/ in Phase 10)


def run_task8_3_reader_upgrade_eval() -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print("  TASK 8.3: UPGRADED WINDOW & N-GRAM SPAN READER EVALUATION RUN", flush=True)
    print("=" * 80, flush=True)

    st_retriever = SentenceTransformerRRFRetriever("all-MiniLM-L6-v2")
    upgraded_reader = FastUpgradedWindowReader(st_retriever)
    results = {}

    for domain in ["finance", "legal"]:
        qa_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain}.json"
        extract_dir = f"external_benchmarks/UniDoc-Bench/extracted_pdfs/{domain}"

        if not os.path.exists(qa_path):
            continue

        with open(qa_path, "r", encoding="utf-8") as f:
            qa_items = json.load(f)

        pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                    glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)

        doc_text_map = {}
        for fpath in pdf_files:
            fname = os.path.basename(fpath)
            if fname.startswith("._"):
                continue
            doc_id = fname.split('.')[0]
            try:
                if fpath.endswith('.pdf'):
                    doc = fitz.open(fpath)
                    full_text = " ".join([page.get_text() for page in doc])
                else:
                    full_text = open(fpath, encoding='utf-8', errors='ignore').read()
                doc_text_map[doc_id] = full_text
            except Exception:
                continue

        st_retriever.index_documents(doc_text_map)

        total_qas = len(qa_items)
        retrieval_hits = 0
        oracle_hits = 0
        true_e2e_hits = 0

        for item in qa_items:
            query = item.get("question") or item.get("rewritten_question_obscured") or ""
            gt_answer = item.get("answer") or item.get("complete_answer") or ""
            gt_doc_ids = parse_unidoc_gt_doc_ids(item)

            top_doc_id, _, _ = st_retriever.retrieve_top_document_rrf(query)
            is_retrieval_hit = (top_doc_id in gt_doc_ids) if gt_doc_ids else False
            if is_retrieval_hit:
                retrieval_hits += 1

            # 1. Upgraded Oracle Extraction
            oracle_ans = ""
            if gt_doc_ids and gt_doc_ids[0] in doc_text_map:
                gt_text = doc_text_map[gt_doc_ids[0]]
                oracle_ans = upgraded_reader.extract_answer_span(query, gt_text)
                if check_answer_match(oracle_ans, gt_answer):
                    oracle_hits += 1

            # 2. Upgraded Pipeline Extraction
            pipeline_ans = ""
            if top_doc_id in doc_text_map:
                retrieved_text = doc_text_map[top_doc_id]
                pipeline_ans = upgraded_reader.extract_answer_span(query, retrieved_text)

            if check_answer_match(pipeline_ans, gt_answer) and is_retrieval_hit:
                true_e2e_hits += 1

        retrieval_acc = retrieval_hits / float(total_qas)
        oracle_acc = oracle_hits / float(total_qas)
        true_e2e_em = true_e2e_hits / float(total_qas)
        cond_reader_acc = (true_e2e_hits / float(retrieval_hits)) if retrieval_hits > 0 else 0.0

        print(f"\n--- TASK 8.3 {domain.upper()} UPGRADED READER METRICS ---", flush=True)
        print(f"  Total Evaluated QA Pairs:      {total_qas}", flush=True)
        print(f"  Document Retrieval Accuracy:   {retrieval_hits} / {total_qas} ({retrieval_acc:.2%})", flush=True)
        print(f"  Oracle-Scoped Answer Accuracy: {oracle_hits} / {total_qas} ({oracle_acc:.2%})", flush=True)
        print(f"  True End-to-End Answer EM:     {true_e2e_hits} / {total_qas} ({true_e2e_em:.2%})", flush=True)
        print(f"  Conditional Reader Success:    {true_e2e_hits} / {retrieval_hits} ({cond_reader_acc:.2%})", flush=True)
        print("-" * 80, flush=True)

        results[domain] = {
            "total_qas": total_qas,
            "retrieval_acc": retrieval_acc,
            "oracle_acc": oracle_acc,
            "true_e2e_em": true_e2e_em,
            "cond_reader_acc": cond_reader_acc
        }

    return results

if __name__ == "__main__":
    run_task8_3_reader_upgrade_eval()
