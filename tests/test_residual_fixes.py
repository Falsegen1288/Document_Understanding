import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.eval_harness_tables import TableEvalHarness

# Enhanced Financial Abbreviations & Domain Keywords
TableEntityTokenizer.ABBREVIATIONS.update({
    'sales': 'revenue',
    'revenues': 'revenue',
    'revenue': 'revenue',
    'earnings': 'income',
    'income': 'income',
    'profit': 'income',
    'wages': 'salaries',
    'salaries': 'salaries',
    'tax': 'tax',
    'taxes': 'tax',
    'basic': 'basic',
    'diluted': 'diluted',
    'fair': 'fair',
    'carrying': 'carrying'
})

def transform_tatqa_doc_enhanced(tat_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = []
    doc_uid = str(tat_doc.get('table', {}).get('uid') or tat_doc.get('doc', {}).get('uid', 'tat_doc_01'))
    tbl_dict = tat_doc.get('table', {})
    
    if isinstance(tbl_dict, dict) and 'table' in tbl_dict:
        raw_grid = tbl_dict['table']
        if raw_grid and len(raw_grid) >= 1:
            # Multi-row header detection
            header_rows_count = 1
            for r_idx in range(1, min(4, len(raw_grid))):
                row_str = ' '.join([str(c) for c in raw_grid[r_idx]])
                if re.search(r'\b(201\d|200\d|202\d|thousands|millions|years ended|carrying|fair value)\b', row_str, re.I):
                    header_rows_count = r_idx + 1

            col_count = max(len(r) for r in raw_grid)
            merged_headers = []
            for c_idx in range(col_count):
                parts = []
                for h_idx in range(header_rows_count):
                    if c_idx < len(raw_grid[h_idx]):
                        val = str(raw_grid[h_idx][c_idx]).strip()
                        if val and val not in parts:
                            parts.append(val)
                merged_headers.append(' :: '.join(parts) if parts else f'Col_{c_idx}')

            rows_raw = raw_grid[header_rows_count:]
            rows = []
            current_section = ""
            for r_idx, row_cells in enumerate(rows_raw):
                row_cells_str = [str(c).strip() for c in row_cells]
                raw_label = row_cells_str[0] if row_cells_str else f'Row_{r_idx}'
                
                # Check if row is a section header (non-first cells are empty)
                non_empty_cells = [c for c in row_cells_str[1:] if c != ""]
                if len(non_empty_cells) == 0 and raw_label != "":
                    current_section = raw_label
                    continue  # Skip inserting empty section header row as data row

                # Prepend current section header to sub-row label if active
                row_label = f"{current_section} :: {raw_label}" if current_section else raw_label
                rows.append({
                    'row_label': row_label,
                    'cell_values': row_cells_str,
                    'cell_bboxes': [[0, 0, 0, 0] for _ in row_cells_str]
                })

            tables.append({
                'table_id': doc_uid,
                'section_path': f'TAT-DQA > {doc_uid}',
                'page': 1,
                'bbox': [0.0, 0.0, 500.0, 500.0],
                'column_headers': merged_headers,
                'rows': rows
            })
    return tables

def main():
    with open("external_benchmarks/TAT-QA/data/tatqa_dataset_dev.json", "r", encoding="utf-8") as f:
        tat_docs = json.load(f)

    index = StrategyARowKVIndex()
    for d in tat_docs:
        tbls = transform_tatqa_doc_enhanced(d)
        if tbls:
            index.ingest_tables(tbls)

    table_only_span_qas = []
    for d in tat_docs:
        doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid'))
        for q in d.get('questions', []):
            if q.get('answer_from') == 'table' and q.get('answer_type') == 'span':
                table_only_span_qas.append((q, doc_uid))

    print(f"Total Table-Only Span Queries: {len(table_only_span_qas)}")

    harness = TableEvalHarness("data/table_qa_eval_dataset.json")

    em_scoped = 0

    for q, doc_uid in table_only_span_qas:
        query = q['question']
        gt = q['answer']
        gt_str = str(gt[0]).strip() if isinstance(gt, list) and gt else str(gt).strip()
        
        tokens = TableEntityTokenizer.tokenize(query)
        cursor = index.conn.cursor()
        val_s = None
        
        if tokens:
            placeholders = ','.join(['?'] * len(tokens))
            sql = f'''
                SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
                FROM entity_index e
                WHERE e.table_id = ? AND e.token IN ({placeholders})
                GROUP BY e.row_id, e.table_id
                ORDER BY hit_count DESC
                LIMIT 1
            '''
            cursor.execute(sql, [doc_uid] + tokens)
            rh = cursor.fetchone()
            if rh:
                row_id = rh['row_id']
                cursor.execute('SELECT cell_id, column_label, raw_value, bbox_json FROM table_cells WHERE row_id = ?', (row_id,))
                cells = cursor.fetchall()
                best_cell, best_score = index._pick_best_matching_cell(cells, tokens)
                val_s = best_cell['raw_value'] if best_cell else None

                if "in which year" in query.lower() or "which year" in query.lower():
                    col_lbl = best_cell['column_label'] if best_cell else ''
                    year_match = re.search(r'\b(201\d|202\d|200\d)\b', col_lbl)
                    if year_match:
                        val_s = year_match.group(1)
                
        is_match = harness._check_exact_match(val_s, gt_str)
        if not is_match:
            clean_s = re.sub(r'[$,\s]', '', str(val_s or '').lower())
            clean_gt = re.sub(r'[$,\s]', '', gt_str.lower())
            if clean_s and clean_gt and clean_s == clean_gt:
                is_match = True

        if is_match:
            em_scoped += 1

    print(f"Corrected Oracle-Scoped Table-Span EM: {em_scoped} / {len(table_only_span_qas)} ({em_scoped/len(table_only_span_qas):.2%})")

if __name__ == "__main__":
    main()
