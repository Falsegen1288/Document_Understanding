import os
import sys
import json
import glob
import time
import re
import fitz # PyMuPDF
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.table_indexing.strategy_a_row_kv import StrategyARowKVIndex
from src.table_indexing.tokenizer import TableEntityTokenizer
from src.routing.dispatch_router import DispatchRouter
from tests.run_unidoc_full_metrics import parse_unidoc_gt_doc_ids

from sentence_transformers import SentenceTransformer

class SentenceTransformerRRFRetriever:
    """
    Production Dense Sentence Transformer + BM25 RRF Hybrid Retriever.
    Dense Branch: sentence-transformers/all-MiniLM-L6-v2 (384-dim dense embeddings).
    Sparse Branch: Okapi BM25 token matching.
    Fusion: Reciprocal Rank Fusion (RRF, k=60).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", k_rrf: int = 60):
        self.k_rrf = k_rrf
        print(f"Loading SentenceTransformer model '{model_name}'...", flush=True)
        self.model = SentenceTransformer(model_name)
        self.doc_text_map = {}
        self.doc_tokens = {}
        self.doc_freqs = defaultdict(int)
        self.doc_embeddings = {}
        self.total_docs = 0
        self.avg_doc_len = 0.0

    def index_documents(self, doc_text_map: Dict[str, str]) -> Dict[str, Any]:
        self.doc_text_map = doc_text_map
        doc_ids = list(doc_text_map.keys())
        texts = [doc_text_map[did][:4000] for did in doc_ids] # First 4000 chars per doc

        t0 = time.time()
        print(f"Encoding {len(texts)} documents with {self.model}...", flush=True)
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        t1 = time.time()

        encoding_time = t1 - t0

        for did, emb, txt in zip(doc_ids, embeddings, texts):
            self.doc_embeddings[did] = emb
            toks = TableEntityTokenizer.tokenize(txt)
            self.doc_tokens[did] = toks
            for t in set(toks):
                self.doc_freqs[t] += 1

        self.total_docs = len(self.doc_tokens)
        total_len = sum(len(toks) for toks in self.doc_tokens.values())
        self.avg_doc_len = total_len / max(1, self.total_docs)

        # Index size estimation: 384 floats (4 bytes each) per doc
        index_bytes = self.total_docs * 384 * 4

        print(f"Dense Indexing Complete in {encoding_time:.2f}s ({len(doc_ids)} docs). Index Size: {index_bytes / 1024:.2f} KB.", flush=True)
        return {
            "encoding_time_sec": encoding_time,
            "index_bytes": index_bytes,
            "total_docs": self.total_docs
        }

    def retrieve_top_document_rrf(self, query: str, k1: float = 1.5, b: float = 0.75) -> Tuple[str, float, float]:
        q_tokens = TableEntityTokenizer.tokenize(query)
        if not q_tokens or not self.doc_tokens:
            return list(self.doc_tokens.keys())[0] if self.doc_tokens else "", 0.0, 0.0

        # 1. Sparse BM25 Ranks
        bm25_scores = {}
        for doc_id, doc_toks in self.doc_tokens.items():
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
            bm25_scores[doc_id] = score

        sorted_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_bm25)}

        # 2. Dense SentenceTransformer Cosine Similarity Ranks
        q_emb = self.model.encode(query, normalize_embeddings=True)
        dense_scores = {}
        for doc_id, d_emb in self.doc_embeddings.items():
            dot = float(sum(q * d for q, d in zip(q_emb, d_emb)))
            dense_scores[doc_id] = dot

        sorted_dense = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)
        dense_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_dense)}

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores = {}
        for doc_id in self.doc_tokens.keys():
            r_bm25 = bm25_ranks.get(doc_id, 1000)
            r_dense = dense_ranks.get(doc_id, 1000)
            rrf_scores[doc_id] = (1.0 / (self.k_rrf + r_bm25)) + (1.0 / (self.k_rrf + r_dense))

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_doc = sorted_rrf[0][0] if sorted_rrf else ""
        return top_doc, bm25_scores.get(top_doc, 0.0), dense_scores.get(top_doc, 0.0)

def check_answer_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    pred_clean = re.sub(r'[^\w\s]', '', str(pred).lower()).strip()
    gt_clean = re.sub(r'[^\w\s]', '', str(gt).lower()).strip()
    if not pred_clean or not gt_clean:
        return False
    if pred_clean == gt_clean or gt_clean in pred_clean or pred_clean in gt_clean:
        return True
    p_toks = set(pred_clean.split())
    g_toks = set(gt_clean.split())
    overlap = p_toks.intersection(g_toks)
    if len(overlap) / float(len(g_toks)) >= 0.5:
        return True
    return False

def run_dense_unidoc_eval(domain_name: str, retriever_instance: SentenceTransformerRRFRetriever) -> Dict[str, Any]:
    qa_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain_name}.json"
    extract_dir = f"external_benchmarks/UniDoc-Bench/extracted_pdfs/{domain_name}"

    if not os.path.exists(qa_path):
        print(f"[{domain_name.upper()}] QA JSON missing: {qa_path}")
        return {}

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)

    index = StrategyARowKVIndex()
    pdf_files = glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True) + \
                glob.glob(os.path.join(extract_dir, "**", "*.txt"), recursive=True)

    print(f"\n================================================================================", flush=True)
    print(f"  TASK 7.2: {domain_name.upper()} DENSE SENTENCE-TRANSFORMER RRF EVALUATION RUN", flush=True)
    print(f"================================================================================", flush=True)
    print(f"Ingesting {len(pdf_files)} {domain_name.capitalize()} PDFs...", flush=True)

    doc_text_map = {}
    all_table_objs = []
    rows_ingested = 0

    for fpath in pdf_files:
        fname = os.path.basename(fpath)
        if fname.startswith("._"):
            continue
        doc_id = fname.split('.')[0]
        try:
            if fpath.endswith('.pdf'):
                doc = fitz.open(fpath)
                full_text = " ".join([page.get_text() for page in doc])
            else:
                full_text = open(fpath, encoding='utf-8', errors='ignore').read()

            doc_text_map[doc_id] = full_text

            lines = [l.strip() for l in full_text.split('\n') if l.strip()]
            rows = []
            for l_idx, line in enumerate(lines[:30]):
                parts = line.split(":", 1)
                h_val = parts[0].strip() if len(parts) > 1 else f"Section_{l_idx}"
                c_val = parts[1].strip() if len(parts) > 1 else line
                rows.append({
                    "row_label": f"{doc_id}_r{l_idx}",
                    "cell_values": [h_val, c_val, doc_id],
                    "cell_bboxes": [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
                })

            all_table_objs.append({
                "table_id": doc_id,
                "section_path": f"UniDoc > {domain_name} > {doc_id}",
                "page": 1,
                "bbox": [0.0, 0.0, 500.0, 500.0],
                "column_headers": ["Section / Header", "Content / Value", "Document ID"],
                "rows": rows
            })
            rows_ingested += len(rows)
        except Exception:
            continue

    index.ingest_tables(all_table_objs)
    perf_meta = retriever_instance.index_documents(doc_text_map)

    router = DispatchRouter(table_index=index, prose_retriever=retriever_instance)

    total_qas = len(qa_items)
    retrieval_hits = 0
    oracle_hits = 0
    true_e2e_hits = 0
    false_positives = 0

    intent_counts = defaultdict(int)

    for item in qa_items:
        query = item.get("question") or item.get("rewritten_question_obscured") or ""
        gt_answer = item.get("answer") or item.get("complete_answer") or ""
        gt_doc_ids = parse_unidoc_gt_doc_ids(item)

        result = router.route_and_execute(query)
        intent = result["intent"]
        pred_ans = result["answer"]
        pred_doc_id = result["doc_id"]

        intent_counts[intent] += 1

        is_retrieval_hit = (pred_doc_id in gt_doc_ids) if gt_doc_ids else False
        if is_retrieval_hit:
            retrieval_hits += 1

        is_ans_match = check_answer_match(pred_ans, gt_answer)

        oracle_match = False
        if gt_doc_ids and gt_doc_ids[0] in doc_text_map:
            oracle_text = doc_text_map[gt_doc_ids[0]]
            oracle_ans, _ = router.execute_prose_extraction(query, [(gt_doc_ids[0], oracle_text)])
            if check_answer_match(oracle_ans, gt_answer):
                oracle_match = True

        if oracle_match:
            oracle_hits += 1

        if is_ans_match:
            if is_retrieval_hit:
                true_e2e_hits += 1
            else:
                false_positives += 1

    retrieval_acc = retrieval_hits / max(1, total_qas)
    oracle_em = oracle_hits / max(1, total_qas)
    true_e2e_em = true_e2e_hits / max(1, total_qas)
    false_pos_rate = false_positives / max(1, total_qas)

    print(f"\n--- [{domain_name.upper()} DENSE SENTENCE-TRANSFORMER METRICS OUTPUT] ---", flush=True)
    print(f"  Query Intent Breakdown:        Table: {intent_counts['table']}, Prose: {intent_counts['prose']}", flush=True)
    print(f"  Dense Encoding Time:           {perf_meta['encoding_time_sec']:.2f}s ({perf_meta['total_docs']} docs)", flush=True)
    print(f"  Dense Index Memory Footprint:  {perf_meta['index_bytes'] / 1024:.2f} KB", flush=True)
    print(f"  Retrieval Accuracy (RRF):      {retrieval_hits} / {total_qas} ({retrieval_acc:.2%})", flush=True)
    print(f"  Oracle-Scoped Answer Accuracy: {oracle_hits} / {total_qas} ({oracle_em:.2%})", flush=True)
    print(f"  True End-to-End Answer EM:     {true_e2e_hits} / {total_qas} ({true_e2e_em:.2%})", flush=True)
    print(f"  False-Positive Rate:           {false_positives} / {total_qas} ({false_pos_rate:.2%})", flush=True)
    print("=" * 80, flush=True)

    return {
        "domain": domain_name,
        "total_qas": total_qas,
        "encoding_time_sec": perf_meta['encoding_time_sec'],
        "index_bytes": perf_meta['index_bytes'],
        "intent_breakdown": dict(intent_counts),
        "retrieval_acc": retrieval_acc,
        "oracle_em": oracle_em,
        "true_e2e_em": true_e2e_em,
        "false_pos_rate": false_pos_rate
    }

if __name__ == "__main__":
    st_retriever = SentenceTransformerRRFRetriever("all-MiniLM-L6-v2")
    fin_res = run_dense_unidoc_eval("finance", st_retriever)
    leg_res = run_dense_unidoc_eval("legal", st_retriever)
