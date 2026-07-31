import os
import sys
import glob
import json
import tarfile
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from tests.adapters.unidoc_adapter import UniDocBenchAdapter
from tests.eval_harness_tables import TableEvalHarness

def extract_and_ingest_domain(domain_name: str, tar_path: str, extract_dir: str, index: StrategyARowKVIndex):
    print(f"\n--- Processing Domain: {domain_name.upper()} ---")
    if not os.path.exists(tar_path):
        print(f"Error: Tarball {tar_path} not found!")
        return None

    print(f"Extracting {tar_path} ({os.path.getsize(tar_path)/(1024*1024):.1f} MB)...")
    domain_extract_path = os.path.join(extract_dir, domain_name)
    os.makedirs(domain_extract_path, exist_ok=True)
    
    start_time = time.time()
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=domain_extract_path)
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

    pdf_files = glob.glob(os.path.join(domain_extract_path, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(domain_extract_path, "**", "*.txt"), recursive=True)
    
    print(f"Extracted {len(pdf_files)} document files for {domain_name}.")

    # Ingest document text/table content into Strategy A's SQLite store
    ingested_tables = 0
    ingested_rows = 0

    for idx, fpath in enumerate(pdf_files):
        doc_id = os.path.basename(fpath)
        # Parse extracted text/tables into structured table schema
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                lines = [line.strip() for line in fp.readlines() if line.strip()]
            
            # Construct a structured table representation per extracted document
            headers = ["Section / Header", "Content / Specification", "Metadata"]
            rows = []
            for l_idx, line in enumerate(lines[:50]):  # Cap at 50 lines per document chunk
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
                "column_headers": headers,
                "rows": rows
            }

            index.ingest_tables([table_obj])
            ingested_tables += 1
            ingested_rows += len(rows)
        except Exception as e:
            continue

    ingest_time = time.time() - start_time
    print(f"Ingestion complete for {domain_name}: {ingested_tables} documents, {ingested_rows} rows indexed in {ingest_time:.2f} seconds.")

    return {
        "domain": domain_name,
        "files_count": len(pdf_files),
        "tables_ingested": ingested_tables,
        "rows_ingested": ingested_rows,
        "ingest_time_sec": round(ingest_time, 2)
    }

def main():
    print("=" * 80)
    print("      PHASE 5: UNIDOC-BENCH REAL PDF INGESTION & SCORING")
    print("=" * 80)

    archives_dir = os.path.join("external_benchmarks", "UniDoc-Bench", "pdf_archives")
    extract_dir = os.path.join("external_benchmarks", "UniDoc-Bench", "extracted_pdfs")
    qa_dir = os.path.join("external_benchmarks", "UniDoc-Bench", "data", "QA", "filtered")

    index = StrategyARowKVIndex()

    # Track domain ingestion stats
    ingest_stats = {}
    domains = ["finance", "legal", "healthcare"]

    for domain in domains:
        tar_path = os.path.join(archives_dir, f"{domain}_pdfs.tar.gz")
        if os.path.exists(tar_path):
            stat = extract_and_ingest_domain(domain, tar_path, extract_dir, index)
            if stat:
                ingest_stats[domain] = stat

    # Confirm SQLite store counts
    cursor = index.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM table_metadata;")
    total_tables = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM table_rows;")
    total_rows = cursor.fetchone()[0]

    print("\n" + "=" * 80)
    print(f"TOTAL POPULATED STRATEGY A SQLITE STORE: {total_tables} Tables, {total_rows} Rows.")
    print("=" * 80)

    # Run UniDoc QA scoring on populated index
    domain_scores = {}

    for domain in domains:
        qa_path = os.path.join(qa_dir, f"{domain}.json")
        if os.path.exists(qa_path) and domain in ingest_stats:
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_items = json.load(f)

            # Categorize QA pairs by modality/type
            text_table_qa = []
            figure_qa = []

            for idx, item in enumerate(qa_items):
                ans_type = item.get("answer_type", "")
                qa_obj = UniDocBenchAdapter.transform_unidoc_qa_pair(item, idx)
                if ans_type == "image_only" or "figure" in item.get("question_type", ""):
                    figure_qa.append(qa_obj)
                else:
                    text_table_qa.append(qa_obj)

            # Score text/table queries against populated index
            scratch_path = f"data/unidoc_real_{domain}.json"
            with open(scratch_path, "w", encoding="utf-8") as f:
                json.dump(text_table_qa, f, indent=2)

            harness = TableEvalHarness(scratch_path)
            metrics = harness.evaluate_index(index, f"UniDoc-Bench Real ({domain})")

            if os.path.exists(scratch_path):
                os.remove(scratch_path)

            domain_scores[domain] = {
                "text_table_count": len(text_table_qa),
                "figure_count": len(figure_qa),
                "metrics": metrics
            }

    print("\n" + "=" * 80)
    print(f"{'Domain':<15} | {'Evaluated QA':<14} | {'Figure (N/A)':<14} | {'Cell EM':<10} | {'Faithfulness':<14} | {'Relevancy':<12} | {'Latency':<8}")
    print("-" * 80)
    for dom, res in domain_scores.items():
        m = res["metrics"]
        print(f"{dom:<15} | {res['text_table_count']:<14} | {res['figure_count']:<14} | {m['cell_exact_match_accuracy']:<10.2%} | {m['ragas_faithfulness']:<14.3f} | {m['ragas_answer_relevancy']:<12.3f} | {m['avg_latency_ms']:<8.2f} ms")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
