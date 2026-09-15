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
from tests.eval_harness_tables import TableEvalHarness
from tests.run_real_unidoc_retrieval import extract_unidoc_evidence_sources, UniDocHybridRetriever

def build_unidoc_production_tables_and_chunks(domain_name: str, index: StrategyARowKVIndex) -> Tuple[Dict[str, str], int, int]:
    extract_dir = f"external_benchmarks/UniDoc-Bench/extracted_pdfs/{domain_name}"
    pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)
    
    doc_text_map = {}
    total_tables_ingested = 0
    total_rows_ingested = 0

    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        doc_id = fname.split('.')[0]
        try:
            lines = []
            if fpath.endswith(".pdf"):
                doc = fitz.open(fpath)
                full_text = " ".join([p.get_text() for p in doc])
                lines = [l.strip() for l in full_text.split('\n') if l.strip()]
            else:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    full_text = fp.read()
                    lines = [l.strip() for l in full_text.split('\n') if l.strip()]
            
            doc_text_map[doc_id] = full_text[:10000]

            # Ingest document table/prose structure into Strategy A SQLite store
            rows = []
            for l_idx, line in enumerate(lines[:40]):
                parts = line.split(":", 1)
                h_val = parts[0].strip() if len(parts) > 1 else f"Paragraph_{l_idx}"
                c_val = parts[1].strip() if len(parts) > 1 else line
                rows.append({
                    "row_label": f"{doc_id}_r{l_idx}",
                    "cell_values": [h_val, c_val, doc_id],
                    "cell_bboxes": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                })

            table_obj = {
                "table_id": doc_id,
                "section_path": f"UniDoc > {domain_name} > {doc_id}",
                "page": 1,
                "bbox": [0.0, 0.0, 500.0, 500.0],
                "column_headers": ["Header / Key", "Content / Value", "Document ID"],
                "rows": rows
            }
            index.ingest_tables([table_obj])
            total_tables_ingested += 1
            total_rows_ingested += len(rows)
        except Exception:
            continue

    return doc_text_map, total_tables_ingested, total_rows_ingested

def evaluate_unidoc_answer_correctness(domain_name: str) -> Dict[str, Any]:
    qa_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain_name}.json"
    if not os.path.exists(qa_path):
        print(f"[{domain_name.upper()}] QA JSON missing: {qa_path}")
        return None

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    # 1. Populate Strategy A SQLite store
    index = StrategyARowKVIndex()
    doc_text_map, tables_ingested, rows_ingested = build_unidoc_production_tables_and_chunks(domain_name, index)

    # 2. Build BM25 Hybrid Retriever
    retriever = UniDocHybridRetriever()
    if doc_text_map:
        retriever.index_documents(doc_text_map)

    # 3. Categorize QA items
    text_table_qas = []
    figure_qas = []

    for item in qa_items:
        ans_type = item.get("answer_type", "")
        text_src, img_src = extract_unidoc_evidence_sources(item)
        if ans_type == "image_only" or (not text_src and img_src):
            figure_qas.append(item)
        else:
            text_table_qas.append((item, text_src))

    total_eval = len(text_table_qas)
    harness = TableEvalHarness("data/table_qa_eval_dataset.json")

    # Metrics
    oracle_correct = 0
    e2e_retrieval_hits = 0
    true_e2e_correct = 0
    value_only_matches = 0
    false_positives = 0
    citations_correct = 0

    trace_samples = []

    for idx, (item, text_src) in enumerate(text_table_qas):
        query = item.get("rewritten_question_obscured", "")
        gt_ans = item.get("complete_answer", "").strip()
        gt_doc_ids = text_src if text_src else []

        # A. Oracle-Scoped Search (using gt_doc_id if present)
        oracle_doc_id = gt_doc_ids[0] if gt_doc_ids else (list(doc_text_map.keys())[0] if doc_text_map else "")
        tokens = TableEntityTokenizer.tokenize(query)
        
        oracle_val = None
        if tokens and oracle_doc_id:
            cursor = index.conn.cursor()
            placeholders = ','.join(['?'] * len(tokens))
            sql = f'''
                SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
                FROM entity_index e
                WHERE e.table_id = ? AND e.token IN ({placeholders})
                GROUP BY e.row_id, e.table_id
                ORDER BY hit_count DESC
                LIMIT 1
            '''
            cursor.execute(sql, [oracle_doc_id] + tokens)
            rh = cursor.fetchone()
            if rh:
                cursor.execute('SELECT raw_value FROM table_cells WHERE row_id = ?', (rh['row_id'],))
                cells = cursor.fetchall()
                if cells:
                    oracle_val = cells[0]['raw_value']

        # Check Oracle Exact/Semantic Match
        is_oracle_match = harness._check_exact_match(oracle_val, gt_ans)
        if not is_oracle_match and oracle_val and gt_ans:
            if any(w.lower() in str(oracle_val).lower() for w in gt_ans.split()[:3] if len(w) > 3):
                is_oracle_match = True

        if is_oracle_match:
            oracle_correct += 1

        # B. End-to-End Hybrid Retrieval + Extraction
        top_retrieved_doc = retriever.retrieve_top_doc(query)
        is_retrieval_hit = top_retrieved_doc in gt_doc_ids if gt_doc_ids else False
        if is_retrieval_hit:
            e2e_retrieval_hits += 1

        e2e_val = None
        if tokens and top_retrieved_doc:
            cursor = index.conn.cursor()
            placeholders = ','.join(['?'] * len(tokens))
            sql = f'''
                SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
                FROM entity_index e
                WHERE e.table_id = ? AND e.token IN ({placeholders})
                GROUP BY e.row_id, e.table_id
                ORDER BY hit_count DESC
                LIMIT 1
            '''
            cursor.execute(sql, [top_retrieved_doc] + tokens)
            rh = cursor.fetchone()
            if rh:
                cursor.execute('SELECT raw_value FROM table_cells WHERE row_id = ?', (rh['row_id'],))
                cells = cursor.fetchall()
                if cells:
                    e2e_val = cells[0]['raw_value']

        is_e2e_val_match = harness._check_exact_match(e2e_val, gt_ans)
        if not is_e2e_val_match and e2e_val and gt_ans:
            if any(w.lower() in str(e2e_val).lower() for w in gt_ans.split()[:3] if len(w) > 3):
                is_e2e_val_match = True

        if is_e2e_val_match:
            value_only_matches += 1
            if is_retrieval_hit:
                true_e2e_correct += 1
                citations_correct += 1
            else:
                false_positives += 1
        elif is_retrieval_hit:
            citations_correct += 1

        if idx < 5:
            trace_samples.append({
                "query": query,
                "gt_ans": gt_ans[:80],
                "gt_docs": gt_doc_ids,
                "retrieved_doc": top_retrieved_doc,
                "retrieval_hit": is_retrieval_hit,
                "oracle_val": oracle_val[:80] if oracle_val else None,
                "e2e_val": e2e_val[:80] if e2e_val else None,
                "oracle_match": is_oracle_match,
                "true_e2e": (is_e2e_val_match and is_retrieval_hit)
            })

    oracle_acc = (oracle_correct / total_eval) if total_eval > 0 else 0.0
    retrieval_acc = (e2e_retrieval_hits / total_eval) if total_eval > 0 else 0.0
    true_e2e_acc = (true_e2e_correct / total_eval) if total_eval > 0 else 0.0
    val_only_rate = (value_only_matches / total_eval) if total_eval > 0 else 0.0
    false_pos_rate = (false_positives / total_eval) if total_eval > 0 else 0.0
    citation_completeness = (citations_correct / total_eval) if total_eval > 0 else 0.0

    print(f"\n--- [{domain_name.upper()} ANSWER-CORRECTNESS RESULTS] ---")
    print(f"  Ingested Corpus Documents:       {tables_ingested} ({rows_ingested} rows)")
    print(f"  Text/Table QA Evaluated:         {total_eval}")
    print(f"  Figure QA (Not Supported):       {len(figure_qas)}")
    print(f"  Oracle-Scoped Answer Accuracy:   {oracle_correct} / {total_eval} ({oracle_acc:.2%})")
    print(f"  Document Retrieval Accuracy:     {e2e_retrieval_hits} / {total_eval} ({retrieval_acc:.2%})")
    print(f"  True End-to-End Answer Accuracy: {true_e2e_correct} / {total_eval} ({true_e2e_acc:.2%})")
    print(f"  Value-Only Match Rate:           {value_only_matches} / {total_eval} ({val_only_rate:.2%})")
    print(f"  False-Positive Rate:             {false_positives} / {total_eval} ({false_pos_rate:.2%})")
    print(f"  Citation Completeness Rate:      {citations_correct} / {total_eval} ({citation_completeness:.2%})")

    return {
        "domain": domain_name,
        "docs_count": tables_ingested,
        "rows_count": rows_ingested,
        "total_qa": len(qa_items),
        "eval_qa": total_eval,
        "figure_qa": len(figure_qas),
        "oracle_acc": oracle_acc,
        "retrieval_acc": retrieval_acc,
        "true_e2e_acc": true_e2e_acc,
        "val_only_rate": val_only_rate,
        "false_pos_rate": false_pos_rate,
        "citation_completeness": citation_completeness,
        "samples": trace_samples
    }

def main():
    print("=" * 80)
    print("      PART B: UNIDOC-BENCH ANSWER-CORRECTNESS CLOSURE & PROVENANCE EVALUATION")
    print("=" * 80)

    for dom in ["finance", "legal"]:
        evaluate_unidoc_answer_correctness(dom)

if __name__ == "__main__":
    main()
