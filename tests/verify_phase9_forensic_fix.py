import os
import sys
import json
import glob
import re
import fitz # PyMuPDF
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.tokenizer import TableEntityTokenizer
from src.routing.dispatch_router import DispatchRouter
from tests.run_unidoc_dense_pipeline import SentenceTransformerRRFRetriever, check_answer_match
from tests.run_unidoc_full_metrics import parse_unidoc_gt_doc_ids
from src.readers.window_reader import FastUpgradedWindowReader

def get_all_gt_chunks(gt_doc_ids: List[str], doc_text_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Forensic Fix: Returns all individual (chunk_id, chunk_text) tuples belonging to any gt_doc_id.
    This allows Oracle evaluation to test each GT chunk at the identical chunk-granularity as E2E retrieval.
    """
    if not gt_doc_ids:
        return []
    
    gt_chunks = []
    for gtd in gt_doc_ids:
        for k, text in doc_text_map.items():
            if k == gtd or k.startswith(gtd + "_") or k.startswith(gtd + ".") or gtd in k:
                gt_chunks.append((k, text))
                
    # Deduplicate by key
    seen = set()
    unique_chunks = []
    for k, text in gt_chunks:
        if k not in seen:
            seen.add(k)
            unique_chunks.append((k, text))
    return unique_chunks

def run_phase9_forensic_audit() -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print("  TASK 9.1: FORENSIC AUDIT — CHUNK-LEVEL ORACLE VS E2E RE-EVALUATION", flush=True)
    print("=" * 80, flush=True)

    st_retriever = SentenceTransformerRRFRetriever("all-MiniLM-L6-v2")
    upgraded_reader = FastUpgradedWindowReader(st_retriever)
    jaccard_router = DispatchRouter(prose_retriever=st_retriever)

    audit_results = {}

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

        # 1. Jaccard Reader Metrics
        j_retrieval_hits = 0
        j_oracle_hits = 0
        j_e2e_hits = 0

        # 2. Upgraded Reader Metrics
        u_retrieval_hits = 0
        u_oracle_hits = 0
        u_e2e_hits = 0

        for item in qa_items:
            query = item.get("question") or item.get("rewritten_question_obscured") or ""
            gt_answer = item.get("answer") or item.get("complete_answer") or ""
            gt_doc_ids = parse_unidoc_gt_doc_ids(item)

            top_doc_id, _, _ = st_retriever.retrieve_top_document_rrf(query)
            
            is_retrieval_hit = False
            if gt_doc_ids:
                is_retrieval_hit = any(top_doc_id == gtd or top_doc_id.startswith(gtd) or gtd in top_doc_id for gtd in gt_doc_ids)

            if is_retrieval_hit:
                j_retrieval_hits += 1
                u_retrieval_hits += 1

            # Fetch ALL GT Chunks
            gt_chunks = get_all_gt_chunks(gt_doc_ids, doc_text_map)
            retrieved_chunk_text = doc_text_map.get(top_doc_id, "")

            # --- A. Jaccard Reader ---
            # 1. Jaccard Oracle Mode (evaluates each GT chunk)
            j_oracle_match = False
            for chunk_id, chunk_text in gt_chunks:
                ans, _ = jaccard_router.execute_prose_extraction(query, [(chunk_id, chunk_text)])
                if check_answer_match(ans, gt_answer):
                    j_oracle_match = True
                    break
            if j_oracle_match:
                j_oracle_hits += 1

            # 2. Jaccard E2E Mode
            j_e2e_ans, _ = jaccard_router.execute_prose_extraction(query, [(top_doc_id, retrieved_chunk_text)]) if retrieved_chunk_text else ("", "")
            if is_retrieval_hit and check_answer_match(j_e2e_ans, gt_answer):
                j_e2e_hits += 1

            # --- B. Upgraded Reader ---
            # 1. Upgraded Oracle Mode (evaluates each GT chunk)
            u_oracle_match = False
            for chunk_id, chunk_text in gt_chunks:
                ans = upgraded_reader.extract_answer_span(query, chunk_text)
                if check_answer_match(ans, gt_answer):
                    u_oracle_match = True
                    break
            if u_oracle_match:
                u_oracle_hits += 1

            # 2. Upgraded E2E Mode
            u_e2e_ans = upgraded_reader.extract_answer_span(query, retrieved_chunk_text) if retrieved_chunk_text else ""
            if is_retrieval_hit and check_answer_match(u_e2e_ans, gt_answer):
                u_e2e_hits += 1

        j_retrieval_acc = j_retrieval_hits / float(total_qas)
        j_oracle_acc = j_oracle_hits / float(total_qas)
        j_e2e_em = j_e2e_hits / float(total_qas)

        u_retrieval_acc = u_retrieval_hits / float(total_qas)
        u_oracle_acc = u_oracle_hits / float(total_qas)
        u_e2e_em = u_e2e_hits / float(total_qas)

        print(f"\n================================================================================", flush=True)
        print(f"  CORRECTED FORENSIC METRICS REPORT: {domain.upper()}", flush=True)
        print(f"================================================================================", flush=True)
        print(f"  Total QA Pairs Evaluated: {total_qas}", flush=True)
        print(f"  Retrieval Accuracy (RRF): {j_retrieval_hits} / {total_qas} ({j_retrieval_acc:.2%})", flush=True)
        print("-" * 80, flush=True)
        print("  1. JACCARD READER (TASK 8.2 RE-EVALUATION):", flush=True)
        print(f"     Oracle-Scoped Answer Acc: {j_oracle_hits} / {total_qas} ({j_oracle_acc:.2%})", flush=True)
        print(f"     True End-to-End Answer EM: {j_e2e_hits} / {total_qas} ({j_e2e_em:.2%})", flush=True)
        print(f"     Sanity Check (True_E2E <= Oracle): [{'PASS' if j_e2e_hits <= j_oracle_hits else 'FAIL'}]", flush=True)
        print("-" * 80, flush=True)
        print("  2. UPGRADED WINDOW READER (TASK 8.3 RE-EVALUATION):", flush=True)
        print(f"     Oracle-Scoped Answer Acc: {u_oracle_hits} / {total_qas} ({u_oracle_acc:.2%})", flush=True)
        print(f"     True End-to-End Answer EM: {u_e2e_hits} / {total_qas} ({u_e2e_em:.2%})", flush=True)
        print(f"     Sanity Check (True_E2E <= Oracle): [{'PASS' if u_e2e_hits <= u_oracle_hits else 'FAIL'}]", flush=True)
        print("=" * 80, flush=True)

        audit_results[domain] = {
            "total_qas": total_qas,
            "retrieval_acc": j_retrieval_acc,
            "jaccard": {"oracle_hits": j_oracle_hits, "oracle_acc": j_oracle_acc, "e2e_hits": j_e2e_hits, "e2e_em": j_e2e_em, "pass": j_e2e_hits <= j_oracle_hits},
            "upgraded": {"oracle_hits": u_oracle_hits, "oracle_acc": u_oracle_acc, "e2e_hits": u_e2e_hits, "e2e_em": u_e2e_em, "pass": u_e2e_hits <= u_oracle_hits}
        }

    return audit_results

if __name__ == "__main__":
    run_phase9_forensic_audit()
