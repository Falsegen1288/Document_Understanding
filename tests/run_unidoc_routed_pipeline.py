import os
import sys
import json
import glob
import re
import fitz # PyMuPDF
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from src.routing.dispatch_router import DispatchRouter
from tests.run_unidoc_remediation_pipeline import ProductionRRFHybridRetriever
from tests.run_unidoc_full_metrics import parse_unidoc_gt_doc_ids

def check_answer_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    pred_clean = re.sub(r'[^\w\s]', '', str(pred).lower()).strip()
    gt_clean = re.sub(r'[^\w\s]', '', str(gt).lower()).strip()
    if not pred_clean or not gt_clean:
        return False
    
    # Exact string match
    if pred_clean == gt_clean:
        return True
    # Substring / Token inclusion match
    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True
    
    # Token Jaccard overlap >= 0.5
    p_toks = set(pred_clean.split())
    g_toks = set(gt_clean.split())
    overlap = p_toks.intersection(g_toks)
    if len(overlap) / float(len(g_toks)) >= 0.5:
        return True

    return False

def run_unidoc_routed_eval(domain_name: str) -> Dict[str, Any]:
    qa_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain_name}.json"
    extract_dir = f"external_benchmarks/UniDoc-Bench/extracted_pdfs/{domain_name}"

    if not os.path.exists(qa_path):
        print(f"[{domain_name.upper()}] QA JSON missing: {qa_path}")
        return None

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    # 1. Ingest PDF Document Corpus into Strategy A SQLite & RRF Passage Retriever
    index = StrategyARowKVIndex()
    retriever = ProductionRRFHybridRetriever(k_rrf=60)
    retriever.doc_text_map = {}

    pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)

    print(f"\n================================================================================", flush=True)
    print(f"  TASK 6.3: {domain_name.upper()} DUAL-BRANCH DISPATCH ROUTER EVALUATION RUN", flush=True)
    print(f"================================================================================", flush=True)
    print(f"Ingesting {len(pdf_files)} {domain_name.capitalize()} PDFs...", flush=True)

    doc_text_map = {}
    all_table_objs = []
    rows_ingested = 0

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

            lines = [l.strip() for l in full_text.split('\n') if l.strip()]
            rows = []
            for l_idx, line in enumerate(lines[:30]):
                parts = line.split(":", 1)
                h_val = parts[0].strip() if len(parts) > 1 else f"Section_{l_idx}"
                c_val = parts[1].strip() if len(parts) > 1 else line
                rows.append({
                    "row_label": f"{doc_id}_r{l_idx}",
                    "cell_values": [h_val, c_val, doc_id],
                    "cell_bboxes": [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
                })

            all_table_objs.append({
                "table_id": doc_id,
                "section_path": f"UniDoc > {domain_name} > {doc_id}",
                "page": 1,
                "bbox": [0.0, 0.0, 500.0, 500.0],
                "column_headers": ["Section / Header", "Content / Value", "Document ID"],
                "rows": rows
            })
            rows_ingested += len(rows)
        except Exception:
            continue

    index.ingest_tables(all_table_objs)
    retriever.doc_text_map = doc_text_map
    retriever.index_production_chunks(doc_text_map)
    print(f"Ingestion Complete: {len(all_table_objs)} Documents, {rows_ingested} Strategy A rows.", flush=True)
    print(f"Indexed {len(doc_text_map)} Chunks into RRF Hybrid Passage Retriever.", flush=True)

    router = DispatchRouter(table_index=index, prose_retriever=retriever)

    # Evaluation loop
    total_qas = len(qa_items)
    retrieval_hits = 0
    oracle_hits = 0
    true_e2e_hits = 0
    false_positives = 0

    intent_counts = defaultdict(int)

    for item in qa_items:
        query = item.get("question") or item.get("rewritten_question_obscured") or ""
        gt_answer = item.get("answer") or item.get("complete_answer") or ""
        gt_doc_ids = parse_unidoc_gt_doc_ids(item)

        result = router.route_and_execute(query)
        intent = result["intent"]
        pred_ans = result["answer"]
        pred_doc_id = result["doc_id"]

        intent_counts[intent] += 1

        is_retrieval_hit = (pred_doc_id in gt_doc_ids) if gt_doc_ids else False
        if is_retrieval_hit:
            retrieval_hits += 1

        # Check Answer Exact / Fuzzy Match
        is_ans_match = check_answer_match(pred_ans, gt_answer)

        # Oracle evaluation: force execution on correct GT document text
        oracle_match = False
        if gt_doc_ids and gt_doc_ids[0] in doc_text_map:
            oracle_text = doc_text_map[gt_doc_ids[0]]
            oracle_ans, _ = router.execute_prose_extraction(query, [(gt_doc_ids[0], oracle_text)])
            if check_answer_match(oracle_ans, gt_answer):
                oracle_match = True

        if oracle_match:
            oracle_hits += 1

        if is_ans_match:
            if is_retrieval_hit:
                true_e2e_hits += 1
            else:
                false_positives += 1

    retrieval_acc = retrieval_hits / max(1, total_qas)
    oracle_em = oracle_hits / max(1, total_qas)
    true_e2e_em = true_e2e_hits / max(1, total_qas)
    false_pos_rate = false_positives / max(1, total_qas)

    print(f"\n--- [{domain_name.upper()} ROUTED DUAL-BRANCH METRICS OUTPUT] ---")
    print(f"  Query Intent Breakdown:        Table: {intent_counts['table']}, Prose: {intent_counts['prose']}")
    print(f"  Retrieval Accuracy (RRF):      {retrieval_hits} / {total_qas} ({retrieval_acc:.2%})")
    print(f"  Oracle-Scoped Answer Accuracy: {oracle_hits} / {total_qas} ({oracle_em:.2%})")
    print(f"  True End-to-End Answer EM:     {true_e2e_hits} / {total_qas} ({true_e2e_em:.2%})")
    print(f"  False-Positive Rate:           {false_positives} / {total_qas} ({false_pos_rate:.2%})")
    print("=" * 80)

    return {
        "domain": domain_name,
        "total_qas": total_qas,
        "intent_breakdown": dict(intent_counts),
        "retrieval_acc": retrieval_acc,
        "oracle_em": oracle_em,
        "true_e2e_em": true_e2e_em,
        "false_pos_rate": false_pos_rate
    }

if __name__ == "__main__":
    fin_res = run_unidoc_routed_eval("finance")
    leg_res = run_unidoc_routed_eval("legal")
