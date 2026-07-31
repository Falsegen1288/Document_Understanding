import sqlite3
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .tokenizer import TableEntityTokenizer

class StrategyARowKVIndex:
    """
    Strategy A: Structured Row-Level Key-Value Index.
    
    Persists structured table grids (from TableFormer/TATR/GLM-OCR) into an in-memory
    relational store (SQLite). Each row is indexed with explicit column header mappings,
    unit normalizations, entity tokens, and bounding box metadata.
    """

    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Create relational schema for tables, rows, cells, and entity lookup index."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS table_metadata (
                table_id TEXT PRIMARY KEY,
                section_path TEXT,
                page INTEGER,
                bbox_json TEXT,
                header_schema_json TEXT
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS table_rows (
                row_id TEXT PRIMARY KEY,
                table_id TEXT,
                row_index INTEGER,
                row_label TEXT,
                row_kv_json TEXT,
                FOREIGN KEY(table_id) REFERENCES table_metadata(table_id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS table_cells (
                cell_id TEXT PRIMARY KEY,
                row_id TEXT,
                table_id TEXT,
                row_label TEXT,
                column_label TEXT,
                raw_value TEXT,
                norm_numeric REAL,
                unit_category TEXT,
                norm_string TEXT,
                bbox_json TEXT,
                FOREIGN KEY(row_id) REFERENCES table_rows(row_id)
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_index (
                token TEXT,
                table_id TEXT,
                row_id TEXT,
                cell_id TEXT
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_token ON entity_index(token);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cell_numeric ON table_cells(unit_category, norm_numeric);")
        self.conn.commit()

    def ingest_tables(self, tables: List[Dict[str, Any]]) -> None:
        """
        Ingest structured tables into relational store.
        Extracts nested headers, normalizes units, and indexes entity tokens.
        """
        cursor = self.conn.cursor()
        
        for table in tables:
            table_id = table["table_id"]
            section_path = table.get("section_path", "")
            page = table.get("page", 1)
            bbox = table.get("bbox", [0, 0, 0, 0])
            column_headers = table.get("column_headers", [])
            
            # 1. Insert Table Metadata
            cursor.execute("""
                INSERT OR REPLACE INTO table_metadata (table_id, section_path, page, bbox_json, header_schema_json)
                VALUES (?, ?, ?, ?, ?)
            """, (table_id, section_path, page, json.dumps(bbox), json.dumps(column_headers)))
            
            rows = table.get("rows", [])
            for r_idx, r in enumerate(rows):
                row_id = f"{table_id}_r{r_idx}"
                row_label = r.get("row_label", f"Row_{r_idx}")
                cell_values = r.get("cell_values", [])
                cell_bboxes = r.get("cell_bboxes", [])  # Optional per-cell bbox
                
                # Build Row KV dictionary mapping column_header -> cell_value
                row_kv = {}
                for c_idx, val in enumerate(cell_values):
                    header = column_headers[c_idx] if c_idx < len(column_headers) else f"Col_{c_idx}"
                    row_kv[header] = val
                    
                    # Compute cell specific bbox (fallback to row/table bbox)
                    c_bbox = cell_bboxes[c_idx] if c_idx < len(cell_bboxes) else bbox
                    
                    # Normalize numerical unit values
                    norm_num, unit_cat, norm_str = TableEntityTokenizer.normalize_unit_value(val)
                    
                    cell_id = f"{row_id}_c{c_idx}"
                    cursor.execute("""
                        INSERT OR REPLACE INTO table_cells 
                        (cell_id, row_id, table_id, row_label, column_label, raw_value, norm_numeric, unit_category, norm_string, bbox_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (cell_id, row_id, table_id, row_label, header, val, norm_num, unit_cat, norm_str, json.dumps(c_bbox)))
                    
                    # Tokenize and index cell content and header for fast exact lookup
                    tokens = TableEntityTokenizer.tokenize(val) + TableEntityTokenizer.tokenize(header)
                    for tok in tokens:
                        cursor.execute("INSERT INTO entity_index (token, table_id, row_id, cell_id) VALUES (?, ?, ?, ?)",
                                       (tok, table_id, row_id, cell_id))

                # Insert Row Record
                cursor.execute("""
                    INSERT OR REPLACE INTO table_rows (row_id, table_id, row_index, row_label, row_kv_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (row_id, table_id, r_idx, row_label, json.dumps(row_kv)))
                
                # Tokenize row label into entity_index
                for tok in TableEntityTokenizer.tokenize(row_label):
                    cursor.execute("INSERT INTO entity_index (token, table_id, row_id, cell_id) VALUES (?, ?, ?, NULL)",
                                   (tok, table_id, row_id))

        self.conn.commit()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Execute structured lookup using:
        1. Exact part number / SKU matching via BM25 entity index
        2. Numerical range predicate filtering (e.g. >= 10mm)
        3. Structural column + row attribute matching
        """
        cursor = self.conn.cursor()
        
        # Check if query contains numerical range predicate (e.g., '>= 10mm', '> 5')
        predicate = TableEntityTokenizer.parse_predicate(query)
        if predicate and predicate['unit_category']:
            return self._query_predicate(predicate, query, top_k)

        # Extract SKU / part numbers or entity tokens
        query_tokens = TableEntityTokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Find matching row_ids by token occurrence count
        placeholders = ",".join(["?"] * len(query_tokens))
        sql = f"""
            SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
            FROM entity_index e
            WHERE e.token IN ({placeholders})
            GROUP BY e.row_id, e.table_id
            ORDER BY hit_count DESC
            LIMIT ?
        """
        cursor.execute(sql, query_tokens + [top_k])
        row_hits = cursor.fetchall()

        results = []
        for rh in row_hits:
            row_id = rh["row_id"]
            table_id = rh["table_id"]
            
            # Fetch table metadata
            cursor.execute("SELECT section_path, page, bbox_json FROM table_metadata WHERE table_id = ?", (table_id,))
            tbl_meta = cursor.fetchone()
            section_path = tbl_meta["section_path"]
            page = tbl_meta["page"]
            table_bbox = json.loads(tbl_meta["bbox_json"])

            # Fetch row and its cells
            cursor.execute("SELECT row_label, row_kv_json FROM table_rows WHERE row_id = ?", (row_id,))
            row_rec = cursor.fetchone()
            row_label = row_rec["row_label"]
            row_kv = json.loads(row_rec["row_kv_json"])

            cursor.execute("SELECT cell_id, column_label, raw_value, bbox_json FROM table_cells WHERE row_id = ?", (row_id,))
            cells = cursor.fetchall()

            # Identify target column/value matching query intent
            best_cell, best_score = self._pick_best_matching_cell(cells, query_tokens)
            target_col = best_cell["column_label"] if best_cell else list(row_kv.keys())[-1]
            target_val = best_cell["raw_value"] if best_cell else list(row_kv.values())[-1]
            cell_bbox = json.loads(best_cell["bbox_json"]) if best_cell else table_bbox

            results.append({
                "score": rh["hit_count"],
                "table_id": table_id,
                "section_path": section_path,
                "page": page,
                "bbox": cell_bbox,
                "extracted_value": target_val,
                "row_label": row_label,
                "column_label": target_col,
                "row_kv": row_kv,
                "retrieved_context": f"Table: {table_id} | Row: {row_label} | {target_col}: {target_val}",
                "citation": {
                    "row_label": row_label,
                    "column_label": target_col,
                    "table_id": table_id,
                    "section_path": section_path,
                    "page": page,
                    "bbox": cell_bbox
                }
            })

        return results

    def _query_predicate(self, predicate: Dict[str, Any], query: str, top_k: int) -> List[Dict[str, Any]]:
        """Query numerical range predicates directly against table_cells index."""
        cursor = self.conn.cursor()
        op = predicate['operator']
        op_sql = "=" if op == "=" else (">=" if op == ">=" else (">" if op == ">" else ("<=" if op == "<=" else "<")))
        unit_cat = predicate['unit_category']
        target_val = predicate['target_val']

        sql = f"""
            SELECT c.cell_id, c.row_id, c.table_id, c.row_label, c.column_label, c.raw_value, c.norm_numeric, c.bbox_json,
                   m.section_path, m.page
            FROM table_cells c
            JOIN table_metadata m ON c.table_id = m.table_id
            WHERE c.unit_category = ? AND c.norm_numeric {op_sql} ?
            ORDER BY c.norm_numeric ASC
            LIMIT ?
        """
        cursor.execute(sql, (unit_cat, target_val, top_k))
        cell_rows = cursor.fetchall()

        results = []
        for cr in cell_rows:
            cell_bbox = json.loads(cr["bbox_json"])
            results.append({
                "score": 10.0,
                "table_id": cr["table_id"],
                "section_path": cr["section_path"],
                "page": cr["page"],
                "bbox": cell_bbox,
                "extracted_value": cr["raw_value"],
                "row_label": cr["row_label"],
                "column_label": cr["column_label"],
                "retrieved_context": f"Table: {cr['table_id']} | Row: {cr['row_label']} | {cr['column_label']}: {cr['raw_value']}",
                "citation": {
                    "row_label": cr["row_label"],
                    "column_label": cr["column_label"],
                    "table_id": cr["table_id"],
                    "section_path": cr["section_path"],
                    "page": cr["page"],
                    "bbox": cell_bbox
                }
            })
        return results

    def _pick_best_matching_cell(self, cells: List[sqlite3.Row], query_tokens: List[str]) -> Tuple[Optional[sqlite3.Row], int]:
        """Select the cell within the matched row that best matches column keywords in query."""
        best_cell = None
        max_score = -1

        q_tokens_set = set(query_tokens)
        
        # Domain keyword boost map
        DOMAIN_KEYWORDS = {
            'temperature', 'temp', 'degc', 'c',
            'voltage', 'volts', 'v',
            'pressure', 'psi',
            'diameter', 'mm',
            'storage', 'tb', 'gb',
            'memory', 'ram',
            'battery', 'wh',
            'material',
            'sterilization',
            'processor', 'cpu',
            'flow', 'cv'
        }

        for c in cells:
            col_tokens = set(TableEntityTokenizer.tokenize(c["column_label"]))
            val_tokens = set(TableEntityTokenizer.tokenize(c["raw_value"]))
            
            # Match query tokens with column label tokens (give 5x weight to column matches)
            col_match = len(q_tokens_set.intersection(col_tokens))
            val_match = len(q_tokens_set.intersection(val_tokens))
            
            # Domain keyword match bonus (+10 points)
            domain_bonus = 0
            matching_domain_toks = q_tokens_set.intersection(col_tokens).intersection(DOMAIN_KEYWORDS)
            if matching_domain_toks:
                domain_bonus = 10 * len(matching_domain_toks)
            
            score = col_match * 5 + val_match + domain_bonus
            if score > max_score:
                max_score = score
                best_cell = c

        return best_cell, max_score
