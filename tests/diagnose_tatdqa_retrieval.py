import os
import sys
import json
import math
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.table_indexing.tokenizer import TableEntityTokenizer

def load_data():
    with open("external_benchmarks/TAT-QA/data/tatqa_dataset_dev.json", "r", encoding="utf-8") as f:
        tat_docs = json.load(f)
    
    table_only_span_qas = []
    for d in tat_docs:
        doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid'))
        for q in d.get('questions', []):
            if q.get('answer_from') == 'table' and q.get('answer_type') == 'span':
                table_only_span_qas.append((q, doc_uid))
    return tat_docs, table_only_span_qas

class RRFTableRetriever:
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        self.doc_tokens = {}
        self.doc_freqs = defaultdict(int)
        self.doc_tfidf_vecs = {}
        self.vocab = {}
        self.total_docs = 0
        self.avg_doc_len = 0.0

    def index(self, tat_docs: List[Dict[str, Any]]):
        for d in tat_docs:
            doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid'))
            tbl_dict = d.get('table', {})
            raw_grid = tbl_dict.get('table', []) if isinstance(tbl_dict, dict) else []
            
            lines = [' '.join([str(c) for c in row if c]) for row in raw_grid]
            grid_text = ' '.join(lines)

            tokens = TableEntityTokenizer.tokenize(grid_text)
            self.doc_tokens[doc_uid] = tokens
            for t in set(tokens):
                self.doc_freqs[t] += 1
                
        self.total_docs = len(self.doc_tokens)
        total_len = sum(len(toks) for toks in self.doc_tokens.values())
        self.avg_doc_len = total_len / max(1, self.total_docs)

        # Build vocabulary & TF-IDF vectors
        all_terms = sorted(list(self.doc_freqs.keys()))
        self.vocab = {term: idx for idx, term in enumerate(all_terms)}
        for doc_uid, doc_toks in self.doc_tokens.items():
            counts = defaultdict(int)
            for t in doc_toks:
                counts[t] += 1
            vec = {}
            norm_sq = 0.0
            for t, tf in counts.items():
                df = self.doc_freqs[t]
                idf = math.log((self.total_docs + 1.0) / (df + 1.0)) + 1.0
                val = (1.0 + math.log(tf)) * idf
                vec[self.vocab[t]] = val
                norm_sq += val * val
            norm = math.sqrt(norm_sq) or 1.0
            self.doc_tfidf_vecs[doc_uid] = {k: v / norm for k, v in vec.items()}

    def retrieve(self, query: str, k1: float = 1.5, b: float = 0.75) -> str:
        q_tokens = TableEntityTokenizer.tokenize(query)
        if not q_tokens:
            return list(self.doc_tokens.keys())[0]

        # 1. BM25 Ranks
        bm25_scores = {}
        for doc_uid, doc_toks in self.doc_tokens.items():
            doc_len = len(doc_toks)
            tok_counts = defaultdict(int)
            for t in doc_toks:
                tok_counts[t] += 1
            score = 0.0
            for qt in set(q_tokens):
                if qt in tok_counts:
                    tf = tok_counts[qt]
                    df = self.doc_freqs.get(qt, 0)
                    idf = max(0.0, float(self.total_docs - df + 0.5) / (df + 0.5))
                    num = tf * (k1 + 1.0)
                    den = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (num / den)
            bm25_scores[doc_uid] = score

        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_bm25)}

        # 2. Dense Vector Cosine Similarity Ranks
        q_counts = defaultdict(int)
        for t in q_tokens:
            if t in self.vocab:
                q_counts[t] += 1
        q_vec = {}
        norm_sq = 0.0
        for t, tf in q_counts.items():
            df = self.doc_freqs[t]
            idf = math.log((self.total_docs + 1.0) / (df + 1.0)) + 1.0
            val = (1.0 + math.log(tf)) * idf
            q_vec[self.vocab[t]] = val
            norm_sq += val * val
        norm = math.sqrt(norm_sq) or 1.0
        q_vec = {k: v / norm for k, v in q_vec.items()}

        dense_scores = {}
        for doc_uid, d_vec in self.doc_tfidf_vecs.items():
            dot = sum(v * d_vec.get(k, 0.0) for k, v in q_vec.items())
            dense_scores[doc_uid] = dot

        sorted_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_dense)}

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores = {}
        for doc_uid in self.doc_tokens.keys():
            r_bm25 = bm25_ranks.get(doc_uid, 1000)
            r_dense = dense_ranks.get(doc_uid, 1000)
            rrf_scores[doc_uid] = (1.0 / (self.k_rrf + r_bm25)) + (1.0 / (self.k_rrf + r_dense))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_rrf[0][0] if sorted_rrf else ""

def evaluate():
    tat_docs, qas = load_data()
    retriever = RRFTableRetriever(k_rrf=60)
    retriever.index(tat_docs)
    hits = 0
    for q, gt_doc_uid in qas:
        pred_doc_uid = retriever.retrieve(q['question'])
        if pred_doc_uid == gt_doc_uid:
            hits += 1
    acc = hits / len(qas)
    print(f"RRF Hybrid Retriever (BM25 + TF-IDF Vector Space RRF k=60): {hits} / {len(qas)} ({acc:.2%})")

if __name__ == "__main__":
    evaluate()
