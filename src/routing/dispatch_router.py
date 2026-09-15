import os
import sys
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer

class DispatchRouter:
    """
    Dual-Branch Query Dispatch Router for Multimodal RAG.
    Classifies incoming document queries as:
      - 'table': Exact Table Cell / Row Lookup -> Routes to Strategy A SQLite Store.
      - 'prose': Conceptual / Unstructured Prose -> Routes to RRF Hybrid Passage Retrieval + Extractive Reader.
    """
    
    TABLE_KEYWORDS = {
        "how much", "what is the amount", "what was the amount", "total", "revenue",
        "costs", "balance", "income", "expense", "net income", "cash flow", "ratio",
        "percentage", "in which year", "which year", "figure", "value", "in thousands",
        "in millions", "table", "row", "column"
    }

    PROSE_KEYWORDS = {
        "explain", "describe", "summary", "purpose", "why", "terms of", "policy",
        "definition", "according to", "clause", "agreement", "contract", "section",
        "context", "main reason", "what does", "condition", "stated in"
    }

    def __init__(self, table_index: StrategyARowKVIndex = None, prose_retriever = None):
        self.table_index = table_index
        self.prose_retriever = prose_retriever

    @classmethod
    def classify_query(cls, query: str) -> str:
        """
        Classifies query intent into 'table' vs 'prose'.
        """
        q_lower = query.lower().strip()
        
        # Explicit table structure keywords
        strict_table_kws = {"table", "column", "row label", "cell value", "grid", "row 0", "column 1"}
        if any(kw in q_lower for kw in strict_table_kws):
            return "table"
        
        # Check for prose question signals
        prose_signals = {"what", "who", "why", "how", "where", "explain", "describe", "according", "stated", "which", "summary", "clause", "agreement"}
        tokens = set(TableEntityTokenizer.tokenize(q_lower))
        
        # If question contains prose question words or is longer than 5 words, route to prose RRF passage retrieval
        if any(t in tokens for t in prose_signals) or len(query.split()) > 5:
            return "prose"
        
        return "table"

    def execute_prose_extraction(self, query: str, top_chunks: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Extractive passage reader over retrieved prose chunks.
        Extracts the most relevant sentence or paragraph containing the answer.
        Returns: (extracted_answer_text, retrieved_doc_id)
        """
        if not top_chunks:
            return "", ""

        best_doc_id, best_text = top_chunks[0]
        q_tokens = set(TableEntityTokenizer.tokenize(query))

        # Split chunk text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', best_text) if len(s.strip()) > 10]
        if not sentences:
            sentences = [best_text[:300]]

        best_sentence = sentences[0]
        best_score = -1.0

        for sent in sentences:
            s_tokens = set(TableEntityTokenizer.tokenize(sent))
            if not s_tokens:
                continue
            intersection = q_tokens.intersection(s_tokens)
            # Jaccard / Overlap score
            score = len(intersection) / float(len(q_tokens) + len(s_tokens) - len(intersection))
            if score > best_score:
                best_score = score
                best_sentence = sent

        return best_sentence, best_doc_id

    def route_and_execute(self, query: str) -> Dict[str, Any]:
        intent = self.classify_query(query)
        
        if intent == "table" and self.table_index:
            # Route to Strategy A SQLite Table Store
            tokens = TableEntityTokenizer.tokenize(query)
            cursor = self.table_index.conn.cursor()
            val_extracted = None
            retrieved_doc_id = ""

            if tokens:
                placeholders = ','.join(['?'] * len(tokens))
                sql = f'''
                    SELECT e.row_id, e.table_id, COUNT(DISTINCT e.token) as hit_count
                    FROM entity_index e
                    WHERE e.token IN ({placeholders})
                    GROUP BY e.row_id, e.table_id
                    ORDER BY hit_count DESC
                    LIMIT 1
                '''
                cursor.execute(sql, tokens)
                rh = cursor.fetchone()
                if rh:
                    retrieved_doc_id = rh['table_id']
                    row_id = rh['row_id']
                    cursor.execute('SELECT cell_id, column_label, raw_value, bbox_json FROM table_cells WHERE row_id = ?', (row_id,))
                    cells = cursor.fetchall()
                    best_cell, _ = self.table_index._pick_best_matching_cell(cells, tokens)
                    val_extracted = best_cell['raw_value'] if best_cell else None

            return {
                "intent": "table",
                "answer": val_extracted or "",
                "doc_id": retrieved_doc_id
            }

        elif self.prose_retriever:
            # Route to Prose RRF Retriever + Extractive Passage Reader
            top_doc_id, bm25_s, dense_s = self.prose_retriever.retrieve_top_document_rrf(query)
            doc_text = self.prose_retriever.doc_tokens.get(top_doc_id, [])
            raw_text = getattr(self.prose_retriever, 'doc_text_map', {}).get(top_doc_id, " ".join(doc_text[:200]))

            extracted_ans, doc_id = self.execute_prose_extraction(query, [(top_doc_id, raw_text)])
            return {
                "intent": "prose",
                "answer": extracted_ans,
                "doc_id": top_doc_id
            }

        return {"intent": intent, "answer": "", "doc_id": ""}
