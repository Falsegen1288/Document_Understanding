import json
from typing import List, Dict, Any

class TATDQAAdapter:
    """
    Adapter mapping TAT-DQA dataset JSON format to Document Understanding indexing schema
    and converting pipeline predictions to official tatqa_eval.py prediction format.
    """

    @staticmethod
    def transform_tatdqa_doc_to_schema(tat_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert a TAT-DQA document JSON representation into Document Understanding
        structured table ingestion schema (used by Strategy A/B/C).
        """
        tables = []
        doc_uid = tat_doc.get("table", {}).get("uid") or tat_doc.get("doc", {}).get("uid", "tat_doc_01")
        tbl_dict = tat_doc.get("table", {})
        
        if isinstance(tbl_dict, dict) and "table" in tbl_dict:
            raw_grid = tbl_dict["table"]
            if raw_grid and len(raw_grid) >= 1:
                headers = [str(c).strip() for c in raw_grid[0]]
                rows_raw = raw_grid[1:] if len(raw_grid) > 1 else raw_grid
                
                rows = []
                for r_idx, row_cells in enumerate(rows_raw):
                    row_cells_str = [str(c).strip() for c in row_cells]
                    row_label = row_cells_str[0] if row_cells_str else f"Row_{r_idx}"
                    rows.append({
                        "row_label": row_label,
                        "cell_values": row_cells_str,
                        "cell_bboxes": [[0, 0, 0, 0] for _ in row_cells_str]
                    })
                    
                tables.append({
                    "table_id": str(doc_uid),
                    "section_path": f"TAT-DQA > {doc_uid}",
                    "page": 1,
                    "bbox": [0.0, 0.0, 500.0, 500.0],
                    "column_headers": headers,
                    "rows": rows
                })
                
        return tables

    @staticmethod
    def format_prediction_for_eval(question_uid: str, extracted_val: str, unit_scale: str = "") -> Dict[str, List[Any]]:
        """
        Format pipeline prediction for official tatqa_eval.py evaluation script:
        { question_uid: [pred_answer, pred_scale] }
        """
        return {
            question_uid: [extracted_val, unit_scale]
        }
