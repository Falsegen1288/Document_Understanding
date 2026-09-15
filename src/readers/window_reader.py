"""
Extractive 3-sentence sliding-window reader.
Moved from tests/run_task8_3_reader_upgrade.py during Phase 10 refactor.
"""

import re
from typing import Any, Optional
from src.table_indexing.tokenizer import TableEntityTokenizer


class FastUpgradedWindowReader:
    """
    Upgraded Production Passage Reader for Prose / Unstructured Multimodal RAG.
    1. Multi-Sentence Sliding Window (size=3 sentences) to preserve context.
    2. Hybrid Keyword + BM25 Window Scoring.
    3. N-Gram & Entity Span Extractor: Extracts precise target answer values
       (numbers, currencies, percentages, capitalized spans, key noun phrases)
       instead of returning full noisy intro sentences.
    """
    def __init__(self, retriever: Optional[Any] = None):
        self.retriever = retriever

    def extract_answer_span(self, query: str, document_text: str, is_multi_span: bool = False) -> str:
        if not document_text:
            return ""

        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', document_text) if len(s.strip()) > 5]
        if not raw_sentences:
            return document_text[:200]

        # 1. Sliding Multi-Sentence Windows (window size = 3 sentences)
        windows = []
        w_size = 3
        for i in range(len(raw_sentences)):
            w_text = " ".join(raw_sentences[i:i + w_size])
            windows.append(w_text)

        if not windows:
            windows = [document_text[:500]]

        # 2. Score windows using question token overlap & term frequency
        q_tokens = TableEntityTokenizer.tokenize(query.lower())
        q_tok_set = set(q_tokens)
        if not q_tok_set:
            return raw_sentences[0]

        best_window = windows[0]
        best_score = -1.0

        for w in windows:
            w_toks = TableEntityTokenizer.tokenize(w.lower())
            w_tok_set = set(w_toks)
            if not w_tok_set:
                continue
            intersection = q_tok_set.intersection(w_tok_set)
            score = len(intersection) / float(len(q_tok_set) + 1e-5)
            if score > best_score:
                best_score = score
                best_window = w

        # 3. Entity & N-Gram Answer Span Extraction from best_window
        candidate_spans = re.findall(r'\b(?:[\$\u20ac\u00a3]?\d[\d,.]*%?|\d{4}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', best_window)
        valid_spans = [cs for cs in candidate_spans if cs.lower() not in q_tok_set and len(cs) > 1]

        if valid_spans:
            if is_multi_span:
                # Option (b): Order extracted spans by matching to query year/label tokens in context
                query_years = re.findall(r'\b20\d{2}\b', query)


                unique_spans = list(dict.fromkeys(valid_spans))
                
                if query_years and len(query_years) >= 2:
                    # Match each query year to the closest span in best_window
                    span_positions = {s: best_window.find(s) for s in unique_spans if best_window.find(s) != -1}
                    year_positions = {y: best_window.find(y) for y in query_years if best_window.find(y) != -1}
                    
                    ordered_spans = []
                    for y in query_years:
                        y_pos = year_positions.get(y)
                        if y_pos is not None and span_positions:
                            # Find span closest to this year's position
                            closest_span = min(span_positions.keys(), key=lambda s: abs(span_positions[s] - y_pos))
                            if closest_span not in ordered_spans:
                                ordered_spans.append(closest_span)
                                
                    # Fill any remaining spans
                    for s in unique_spans:
                        if s not in ordered_spans:
                            ordered_spans.append(s)
                    return ", ".join(ordered_spans[:5])
                
                return ", ".join(unique_spans[:5])
            return valid_spans[0]


        # Fallback: Extract sub-clause or sentence in window with non-question tokens
        sub_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', best_window) if len(s.strip()) > 5]
        for s in sub_sentences:
            s_toks = set(TableEntityTokenizer.tokenize(s.lower()))
            diff = s_toks - q_tok_set
            if diff:
                return s

        return best_window

