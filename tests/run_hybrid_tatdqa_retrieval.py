import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from tests.test_residual_fixes import transform_tatqa_doc_enhanced
from tests.eval_harness_tables import TableEvalHarness

class TATDQAHybridTableRetriever:
    """
    Production RRF Hybrid Table Retriever for TAT-DQA.
    Fuses sparse Okapi BM25 token matching with high-dimensional TF-IDF vector space
    cosine similarity via Reciprocal Rank Fusion (RRF, k=60):
    RRF(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_dense(d))
    """
    def __init__(self, k_rrf: int = 60):
        self.k_rrf = k_rrf
        self.doc_chunks = {}
        self.doc_tokens = {}
        self.doc_freqs = defaultdict(int)
        self.doc_tfidf_vecs = {}
        self.vocab = {}
        self.total_docs = 0
        self.avg_doc_len = 0.0

    def index_tables(self, tat_docs: List[Dict[str, Any]]):
        for d in tat_docs:
            doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid', 'tat_doc_01'))
            tbl_dict = d.get('table', {})
            raw_grid = tbl_dict.get('table', []) if isinstance(tbl_dict, dict) else []
            
            lines = [' '.join([str(c) for c in row if c]) for row in raw_grid]
            grid_text = ' '.join(lines)
            tokens = TableEntityTokenizer.tokenize(grid_text)
            
            self.doc_chunks[doc_uid] = grid_text
            self.doc_tokens[doc_uid] = tokens
            for t in set(tokens):
                self.doc_freqs[t] += 1
                
        self.total_docs = len(self.doc_chunks)
        total_len = sum(len(toks) for toks in self.doc_tokens.values())
        self.avg_doc_len = total_len / max(1, self.total_docs)

        # Build vocabulary & TF-IDF vectors
        import math
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

    def retrieve_top_table(self, query: str, k1: float = 1.5, b: float = 0.75) -> str:
        q_tokens = TableEntityTokenizer.tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return list(self.doc_chunks.keys())[0] if self.doc_chunks else ""

        # 1. Sparse BM25 Ranks
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

        # 2. Dense TF-IDF Cosine Similarity Ranks
        import math
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

def main():
    print("=" * 80)
    print("      PART A: TAT-DQA REAL HYBRID RETRIEVAL & CITATION-AWARE EVALUATION")
    print("=" * 80)

    with open("external_benchmarks/TAT-QA/data/tatqa_dataset_dev.json", "r", encoding="utf-8") as f:
        tat_docs = json.load(f)

    # 1. Index tables into Hybrid Table Retriever
    retriever = TATDQAHybridTableRetriever()
    retriever.index_tables(tat_docs)
    print(f"Indexed {retriever.total_docs} tables into Hybrid BM25 Table Retriever.")

    # 2. Ingest tables into Strategy A SQLite Store (Post-Fix Engine)
    index = StrategyARowKVIndex()
    for d in tat_docs:
        tbls = transform_tatqa_doc_enhanced(d)
        if tbls:
            index.ingest_tables(tbls)

    # 3. Extract table-only span queries
    table_only_span_qas = []
    for d in tat_docs:
        doc_uid = str(d.get('table', {}).get('uid') or d.get('doc', {}).get('uid'))
        for q in d.get('questions', []):
            if q.get('answer_from') == 'table' and q.get('answer_type') == 'span':
                table_only_span_qas.append((q, doc_uid))

    total_queries = len(table_only_span_qas)
    print(f"Loaded {total_queries} Table-Only Span Queries.")

    harness = TableEvalHarness("data/table_qa_eval_dataset.json")

    # Metrics Tracking
    retrieval_hits = 0
    true_e2e_em = 0
    value_only_matches = 0
    false_positives = 0

    trace_records = []

    for idx, (q, gt_doc_uid) in enumerate(table_only_span_qas):
        query = q['question']
        gt = q['answer']
        gt_str = str(gt[0]).strip() if isinstance(gt, list) and gt else str(gt).strip()

        # Step A.1: Hybrid Table Retrieval (Zero access to gt_doc_uid!)
        retrieved_doc_uid = retriever.retrieve_top_table(query)
        is_retrieval_hit = (retrieved_doc_uid == gt_doc_uid)
        if is_retrieval_hit:
            retrieval_hits += 1

        # Step A.2: Strategy A Extraction from Retrieved Table
        tokens = TableEntityTokenizer.tokenize(query)
        cursor = index.conn.cursor()
        val_extracted = None
        
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
            cursor.execute(sql, [retrieved_doc_uid] + tokens)
            rh = cursor.fetchone()
            if rh:
                row_id = rh['row_id']
                cursor.execute('SELECT cell_id, column_label, raw_value, bbox_json FROM table_cells WHERE row_id = ?', (row_id,))
                cells = cursor.fetchall()
                best_cell, best_score = index._pick_best_matching_cell(cells, tokens)
                val_extracted = best_cell['raw_value'] if best_cell else None

                if "in which year" in query.lower() or "which year" in query.lower():
                    col_lbl = best_cell['column_label'] if best_cell else ''
                    year_match = re.search(r'\b(201\d|202\d|200\d)\b', col_lbl)
                    if year_match:
                        val_extracted = year_match.group(1)

        # Check Value Exact Match
        is_val_match = harness._check_exact_match(val_extracted, gt_str)
        if not is_val_match:
            clean_s = re.sub(r'[$,\s]', '', str(val_extracted or '').lower())
            clean_gt = re.sub(r'[$,\s]', '', gt_str.lower())
            if clean_s and clean_gt and clean_s == clean_gt:
                is_val_match = True

        if is_val_match:
            value_only_matches += 1
            if is_retrieval_hit:
                true_e2e_em += 1
            else:
                false_positives += 1

        if idx < 10:
            trace_records.append({
                "q_uid": q['uid'],
                "query": query,
                "gt_str": gt_str,
                "gt_doc_uid": gt_doc_uid,
                "retrieved_doc_uid": retrieved_doc_uid,
                "retrieval_hit": is_retrieval_hit,
                "extracted_val": val_extracted,
                "val_match": is_val_match,
                "true_e2e": (is_val_match and is_retrieval_hit)
            })

    # Summary Statistics
    retrieval_accuracy = retrieval_hits / total_queries
    true_e2e_rate = true_e2e_em / total_queries
    val_only_rate = value_only_matches / total_queries
    false_pos_rate = false_positives / total_queries

    print("\n" + "=" * 80)
    print(f"TAT-DQA REAL HYBRID RETRIEVAL & CITATION-AWARE SCORING RESULTS ({total_queries} QUERIES)")
    print("-" * 80)
    print(f"  Table Retrieval Accuracy:     {retrieval_hits} / {total_queries} ({retrieval_accuracy:.2%})")
    print(f"  True End-to-End Table EM:     {true_e2e_em} / {total_queries} ({true_e2e_rate:.2%})")
    print(f"  Value-Only Match Rate:        {value_only_matches} / {total_queries} ({val_only_rate:.2%})")
    print(f"  False-Positive Rate:          {false_positives} / {total_queries} ({false_pos_rate:.2%})")
    print(f"  Oracle-Scoped Baseline (Ref): 47 / {total_queries} (28.14%)")
    print("=" * 80 + "\n")

    print("Sample Traces:")
    print(json.dumps(trace_records[:5], indent=2))

if __name__ == "__main__":
    main()
