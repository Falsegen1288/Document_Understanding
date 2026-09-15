import os
import sys
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.adapters.tatdqa_adapter import TATDQAAdapter
from tests.eval_harness_tables import TableEvalHarness

# Add financial abbreviations
TableEntityTokenizer.ABBREVIATIONS.update({
    'sales': 'revenue',
    'revenues': 'revenue',
    'revenue': 'revenue',
    'earnings': 'income',
    'income': 'income',
    'profit': 'income',
    'wages': 'salaries',
    'salaries': 'salaries',
})

def transform_tatqa_doc_with_header_merging(tat_doc):
    tables = []
    doc_uid = str(tat_doc.get('table', {}).get('uid') or tat_doc.get('doc', {}).get('uid', 'tat_doc_01'))
    tbl_dict = tat_doc.get('table', {})
    
    if isinstance(tbl_dict, dict) and 'table' in tbl_dict:
        raw_grid = tbl_dict['table']
        if raw_grid and len(raw_grid) >= 1:
            header_rows_count = 1
            for r_idx in range(1, min(4, len(raw_grid))):
                row_str = ' '.join([str(c) for c in raw_grid[r_idx]])
                if re.search(r'\b(201\d|200\d|202\d|thousands|millions|years ended)\b', row_str, re.I):
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
            for r_idx, row_cells in enumerate(rows_raw):
                row_cells_str = [str(c).strip() for c in row_cells]
                row_label = row_cells_str[0] if row_cells_str else f'Row_{r_idx}'
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
        tbls = transform_tatqa_doc_with_header_merging(d)
        if tbls:
            index.ingest_tables(tbls)

    table_only_span_qas = []
    for d in tat_docs:
        doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid'))
        for q in d.get('questions', []):
            if q.get('answer_from') == 'table' and q.get('answer_type') == 'span':
                table_only_span_qas.append((q, doc_uid, d.get('table', {}).get('table')))

    harness = TableEvalHarness('data/table_qa_eval_dataset.json')

    oracle_present_failures = []

    for q, doc_uid, raw_grid in table_only_span_qas:
        query = q['question']
        gt = q['answer']
        gt_scale = q.get('scale', '')
        gt_str = str(gt[0]).strip() if isinstance(gt, list) and gt else str(gt).strip()

        # Check if GT answer is present in target document table
        cursor = index.conn.cursor()
        sql = '''
            SELECT r.row_id, r.row_label, r.row_kv_json
            FROM table_rows r
            WHERE r.table_id = ?
        '''
        cursor.execute(sql, (doc_uid,))
        rows = cursor.fetchall()

        clean_gt = re.sub(r'[$,\s]', '', gt_str.lower())
        in_table = False
        target_cell_val = None
        target_row_lbl = None
        target_col_lbl = None

        for r in rows:
            kv = json.loads(r['row_kv_json'])
            for col_k, cell_v in kv.items():
                clean_v = re.sub(r'[$,\s]', '', str(cell_v).lower())
                if clean_gt and clean_v and (clean_gt == clean_v or clean_gt in clean_v):
                    in_table = True
                    target_cell_val = cell_v
                    target_row_lbl = r['row_label']
                    target_col_lbl = col_k
                    break
            if in_table:
                break

        if not in_table:
            continue  # Only trace queries where target cell IS present in table!

        # Now run Oracle-Scoped search on this query
        tokens = TableEntityTokenizer.tokenize(query)
        val_extracted = None
        row_extracted = None
        col_extracted = None

        if tokens:
            placeholders = ','.join(['?'] * len(tokens))
            sql_search = f'''
                SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
                FROM entity_index e
                WHERE e.table_id = ? AND e.token IN ({placeholders})
                GROUP BY e.row_id, e.table_id
                ORDER BY hit_count DESC
                LIMIT 1
            '''
            cursor.execute(sql_search, [doc_uid] + tokens)
            rh = cursor.fetchone()
            if rh:
                r_id = rh['row_id']
                cursor.execute('SELECT cell_id, column_label, raw_value, row_label FROM table_cells WHERE row_id = ?', (r_id,))
                cells = cursor.fetchall()
                best_cell, best_score = index._pick_best_matching_cell(cells, tokens)
                if best_cell:
                    val_extracted = best_cell['raw_value']
                    row_extracted = best_cell['row_label']
                    col_extracted = best_cell['column_label']

                    if "in which year" in query.lower() or "which year" in query.lower():
                        year_match = re.search(r'\b(201\d|202\d|200\d)\b', col_extracted)
                        if year_match:
                            val_extracted = year_match.group(1)

        is_em = harness._check_exact_match(val_extracted, gt_str)
        if not is_em:
            clean_ext = re.sub(r'[$,\s]', '', str(val_extracted or '').lower())
            if clean_ext and clean_gt and clean_ext == clean_gt:
                is_em = True

        if not is_em:
            oracle_present_failures.append({
                "q_uid": q['uid'],
                "query": query,
                "gt": gt_str,
                "gt_scale": gt_scale,
                "doc_uid": doc_uid,
                "target_row_lbl": target_row_lbl,
                "target_col_lbl": target_col_lbl,
                "target_cell_val": target_cell_val,
                "extracted_val": val_extracted,
                "extracted_row_lbl": row_extracted,
                "extracted_col_lbl": col_extracted,
                "raw_grid_sample": raw_grid[:3] if raw_grid else []
            })

    print("=" * 80)
    print(f"ISOLATED {len(oracle_present_failures)} ORACLE-PRESENT FAILING QUERIES")
    print("=" * 80)

    # Print 15 sample traces
    sample_15 = oracle_present_failures[:15]
    print(json.dumps(sample_15, indent=2))

if __name__ == "__main__":
    main()
