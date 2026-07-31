import json
from typing import List, Dict, Any

class UniDocBenchAdapter:
    """
    Adapter mapping UniDoc-Bench dataset JSON format to Document Understanding
    evaluation harness schema.
    """

    @staticmethod
    def transform_unidoc_qa_pair(unidoc_item: Dict[str, Any], index_idx: int) -> Dict[str, Any]:
        """
        Convert a UniDoc-Bench QA record (from healthcare.json, finance.json, etc.)
        to our evaluation harness schema (data/table_qa_eval_dataset.json compatible).
        """
        query = unidoc_item.get("rewritten_question_obscured", "")
        ground_truth_answer = unidoc_item.get("complete_answer", "")
        q_type = unidoc_item.get("question_type", "unidoc_multimodal")
        ans_type = unidoc_item.get("answer_type", "text")
        
        # Extract primary evidence source chunk/img path
        chunks = unidoc_item.get("chunk_used", {})
        evidence_src = "unidoc_source"
        for c_key, c_val in chunks.items():
            if c_val.get("used"):
                meta = c_val.get("metadata")
                if isinstance(meta, dict):
                    evidence_src = meta.get("source", evidence_src)
                elif isinstance(meta, str):
                    evidence_src = meta
                break

        return {
            "query_id": f"unidoc_qa_{index_idx:04d}",
            "query_type": f"unidoc_{q_type}",
            "query": query,
            "ground_truth_value": ground_truth_answer,
            "ground_truth_citation": {
                "row_label": "UniDoc Evidence Chunk",
                "column_label": ans_type,
                "table_id": evidence_src,
                "section_path": f"UniDoc-Bench > {evidence_src}",
                "page": 1,
                "bbox": [0.0, 0.0, 100.0, 100.0]
            }
        }
