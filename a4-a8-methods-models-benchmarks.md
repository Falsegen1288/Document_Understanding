# A4–A8: Methods, Algorithms, Models & Public Benchmarks

For each stage: the candidate methods, the public benchmarks used to judge them, and the
GitHub repos/tools to start from. No model/framework picked yet — that's the next step,
once you've reviewed this.

---

## A4 — Chunking

### Methods / Algorithms
| Method | What it does | When it wins |
|---|---|---|
| Fixed-size / character splitting | Cuts every N characters, ignores structure | Prototyping only |
| Recursive character splitting | Tries paragraph → sentence → word boundaries in order, falls back gracefully | Best general-purpose default — <cite index="34-1">recursive splitting achieves the highest end-to-end accuracy in benchmarks and handles mixed document types well</cite> |
| Sentence-based | Splits strictly on sentence boundaries | Short-form Q&A content |
| Semantic chunking | Groups sentences by embedding similarity into thematically coherent chunks | <cite index="36-1">Best accuracy in isolation — up to ~70% lift vs naive baselines in some benchmarks</cite>, but <cite index="34-1">can produce chunks that retrieve well but lack enough context for the LLM to generate accurate answers</cite> |
| Page-level chunking | One chunk per page | <cite index="35-1">Won NVIDIA's 2024 benchmark (0.648 accuracy, lowest variance) but only for paginated documents</cite> |
| Hierarchical / parent-child ("small-to-big") | Small chunk for matching, larger parent for LLM context | Good middle ground — precise retrieval + full context at generation |
| Late chunking | Embed the full document first, then pool into chunks — preserves cross-chunk context (pronouns, cross-references) | <cite index="36-1">Use when chunks are ambiguous without surrounding context (headers, pronouns, cross-references)</cite> |
| Contextual retrieval (Anthropic's method) | Prepend an LLM-generated short context/summary to each chunk before embedding, so the chunk is self-contained | Best when chunks lose meaning in isolation (your case — scientific/medical docs with heavy cross-referencing) |
| Agentic / LLM-based chunking | An LLM decides chunk boundaries directly based on meaning | Highest cost, use selectively on high-value documents only |

### Public Benchmarks
- **Vecta/FloTorch end-to-end chunking benchmark** — compares strategies on real retrieval tasks, distinguishing retrieval recall from end-to-end generation accuracy; <cite index="34-1">a critical finding is that optimizing for recall alone can hurt the generation step, since retrieval recall and end-to-end accuracy are different metrics</cite>.
- **NVIDIA 2024 chunking benchmark** (page-level results above).
- Chroma's chunking-strategy evaluation research (open methodology, worth replicating on your own corpus).

### GitHub / Tools
- `langchain` — `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`, `SemanticChunker`
- `llama-index` — Node Parsers (multiple strategies, well-documented)
- GitHub topic `semantic-chunking` — several standalone implementations worth referencing
- Your own repo already handles A2/A3 — chunking logic should consume its `section_path` output directly for structure-aware splitting

**Given your corpus (scientific papers + medical docs, heavy cross-referencing, tables):** recursive/structure-aware chunking as the base, contextual retrieval prepended per chunk, and hierarchical parent-child for tables/figures (small chunk = table alone, parent = surrounding section).

---

## A5 — Embedding Generation

### Methods / Algorithms
- **Dense bi-encoder embeddings** — standard single-vector semantic embeddings (most models below).
- **Sparse embeddings (SPLADE, BM25)** — keyword-weighted, complements dense for exact-term matches (product codes, drug names, author names — relevant for your medical/scientific corpus).
- **Multi-vector / late-interaction (ColBERT)** — token-level embeddings, richer matching, heavier storage.
- **Matryoshka representation learning** — <cite index="41-1">lets you reduce embedding dimensions with minimal quality loss, cutting storage costs</cite>, supported by several 2026 models.
- **Domain fine-tuning** — <cite index="41-1">fine-tuning shows +10-30% gains for specialized domains like legal and medical</cite>, directly relevant given your medical document corpus.

### Candidate Models
| Model | Notes |
|---|---|
| BGE-M3 | <cite index="45-1">MIT-licensed workhorse covering 100+ languages with dense, sparse, and multi-vector retrieval in one model — most production RAG stacks default here</cite> |
| Qwen3-Embedding-8B | <cite index="45-1">Open-source, state-of-the-art with Q4 quantization at ~5GB memory</cite>; <cite index="46-1">Apache 2.0 licensed</cite> |
| NV-Embed-v2 | <cite index="45-1">Leads MTEB where licensing permits</cite> |
| Gemini Embedding | <cite index="44-1">Leads English MTEB at 68.32</cite> |
| Cohere embed-v4 / Voyage-3-large / voyage-4 | Strong commercial API options, multimodal support in embed-v4 |
| nomic-embed-text | <cite index="45-1">Best size/quality balance for lightweight local deployment, 8,192 token context</cite> |
| Granite-vision embedding / Qwen3-VL | <cite index="45-1">For multimodal RAG over PDFs and images</cite> — relevant given your figure/table extraction |

### Public Benchmarks
- **MTEB / MMTEB** (Massive Text Embedding Benchmark) — <cite index="42-1">the standard public embedding leaderboard covering retrieval, classification, clustering, reranking, semantic similarity, and multilingual search; use it as a shortlist, then choose by latency, dimensions, context, and license</cite>.
- **BEIR** — <cite index="64-1">18 diverse datasets including MS MARCO, Natural Questions, HotpotQA, and domain-specific corpora spanning biomedical, legal, scientific, and technical content — a model that scores well here generalizes to out-of-distribution queries</cite>, directly relevant to your mixed medical/scientific corpus.
- The single most important line from all sources: <cite index="47-1">no benchmark can fully capture the nuances of a specific dataset — document style, query phrasing, and domain vocabulary all interact in ways that shape retrieval quality, so the decisive factor should always be how a model performs on your own corpus</cite>.

### GitHub / Tools
- `FlagEmbedding` (BAAI/BGE models)
- `sentence-transformers`
- MTEB repo (`embeddings-benchmark/mteb`) — run it yourself on a domain sample

---

## A6 — Metadata Enrichment & Access-Control Tagging

This stage has fewer "algorithms" and more architectural patterns:
- **RBAC/ABAC tagging at ingestion** — attach role- or attribute-based permission tags per chunk at write time (matches your A1 design decision).
- **Metadata schema design** — document type, date, department, confidentiality level, language — all become filterable fields at query time.
- **Real-time IAM sync patterns** — webhook/event-driven permission updates from Okta/Azure AD/Auth0 rather than batch resync, so revocation is immediate.

No public benchmark applies here directly — this is evaluated via **security audit / access-control testing** rather than accuracy metrics. Worth listing as a checklist item in your eval, not a leaderboard.

---

## A7 — Storage (Vector DB + Keyword Index)

### Methods / Algorithms
- **ANN algorithms**: HNSW (most common), IVF, DiskANN — determine the index's speed/recall/memory tradeoff.
- **Hybrid search**: dense + BM25/sparse fused via **Reciprocal Rank Fusion (RRF)** — <cite index="59-1">RRF formula: score(d) = Σ 1/(k + rank_in_list_i(d)) for each retrieval method, with k=60 as standard — this is the pattern used by Cohere, OpenAI's RAG cookbooks, and most production systems</cite>.
- **Metadata filtering** — pre- or post-filter by access/date/type before or after ANN search (pre-filter is usually more efficient at scale).

### Candidate Databases
| DB | Notes |
|---|---|
| Qdrant | <cite index="49-1">Leads open-source speed — 10-25% faster than Weaviate or Milvus on common workloads, thanks to its Rust implementation</cite> |
| Weaviate | <cite index="53-1">Native BM25 + vector hybrid — the most architecturally coherent option if retrieval quality depends on fusing keyword and semantic signals</cite> |
| Milvus | <cite index="51-1">Serious distributed system for very large scale, needs k8s and dedicated ops</cite> |
| pgvector | <cite index="49-1">Right default if the workload is under 10M vectors and the team already runs Postgres</cite> — same backups/access-control tooling as the rest of your stack |
| LanceDB | <cite index="51-1">Embedded-first, zero-ops for some use cases, integrates well with data-lake architectures</cite> |

### Public Benchmarks
- **VectorDBBench** — <cite index="50-1">an open-source tool testing end-to-end database performance including client overhead, filtering, and concurrent queries, rather than algorithm-level metrics alone</cite>.
- **ANN-Benchmarks** — standard raw algorithm-level recall/latency comparison.
- Note the caveat every source repeats: <cite index="50-1">actual performance varies with hardware, software version, dataset distribution, index parameters, and concurrency — run your own benchmarks with representative data before a production decision</cite>.

### GitHub / Tools
- `zilliztech/VectorDBBench`
- `erikbern/ann-benchmarks`

---

## A8 — Ground Truth Generation, Evaluation Benchmarks & Comparison

(You already have the detailed elaboration of this from before — here's the methods/benchmark layer underneath it.)

### GT QA Generation Methods
- **LLM-based synthetic QA generation** (what you're doing — Claude, human-reviewed).
- **RAGAS's built-in testset generation module** — an alternative/supplement, generates synthetic query-context-answer triples automatically with configurable question-type distributions (simple, reasoning, multi-context, conditional).

### Evaluation Frameworks (already chosen: RAGAS, DeepEval, TruLens)
Confirmed division of labor: <cite index="24-1">use DeepEval for automated test gates and agent testing; use RAGAS for focused RAG retrieval and generation evaluation; TruLens is the third option, best for continuous production monitoring with its Triad metrics</cite>. <cite index="29-1">The pattern that works: RAGAS for fast iteration during development, DeepEval as a CI gate, TruLens for production monitoring — each earns its place, and combining them costs little while covering the full lifecycle</cite>.

Known pitfalls to design around: <cite index="29-1">judge bias (an LLM judge from the same provider as the model being evaluated is too forgiving — use a different family for judging), ground-truth drift (labeled test sets go stale as products change — refresh quarterly), and single-score blindness (a 90% average can hide a 60% score on the most-important question class)</cite>.

### Public Benchmark Datasets (to supplement your own golden set, or sanity-check your eval methodology against known-hard cases)
| Benchmark | Focus |
|---|---|
| **BEIR** | General retrieval quality across domains including biomedical/scientific |
| **HotpotQA** | <cite index="71-1">113K questions requiring two-hop reasoning, with annotated supporting facts</cite> — good stress test for your multi-hop question type |
| **MuSiQue** | <cite index="72-1">Extends to 4-hop reasoning with decomposed annotations</cite> — harder multi-hop stress test |
| **2WikiMultiHopQA** | <cite index="70-1">Leverages structured Wikipedia relationships for multi-hop reasoning</cite> |
| **RGB (Retrieval Augmented Generation Benchmark)** | <cite index="66-1">One of the earliest RAG-specific benchmarks, assessing simple retrieve-and-answer cases</cite> |
| **CRAG (Comprehensive RAG Benchmark)** | Broader, more realistic RAG task mix (Meta/NeurIPS 2024) |
| **MultiHop-RAG** | <cite index="67-1">Explicit annotation of reasoning chains, hop counts, and semantic retrieval difficulty</cite> — directly useful for testing your multi-hop question_type slice |
| **MTRAG** | <cite index="68-1">A human-generated multi-turn RAG benchmark across 4 domains — useful if you ever extend to conversational/multi-turn queries</cite> |
| **LegalBench-RAG / domain-specific variants** | Template for building your own medical-domain-specific benchmark the same way |

These aren't a replacement for your Claude-generated + human-reviewed golden set — they're useful as an **external sanity check**: if your pipeline does well on your own golden set but poorly on a known public multi-hop benchmark in a similar domain, that's a signal your golden set isn't stressing the pipeline hard enough.

### GitHub / Tools
- `explodinggradients/ragas`
- `confident-ai/deepeval`
- `truera/trulens`
- `beir-cellar/beir`

---

## Retrieval & Reranking (needed to run A8 evaluations, full detail comes later in the B-phase)
Since A8 evaluation requires running actual retrieval to produce `retrieved_contexts`, worth flagging now:
- **Hybrid retrieval + RRF fusion**, as above.
- **Reranking** — <cite index="57-1">the common pattern is bi-encoder retrieval to top-50 or top-100 followed by reranking, gated by corpus-specific eval lift and latency budget</cite>. Candidates: <cite index="57-1">Cohere Rerank 4 for lowest-friction managed path, BGE Reranker v2-m3 for open-license self-host, ColBERTv2 when token-level interaction matters more than the cross-encoder ceiling</cite>. <cite index="60-1">Qwen3-Reranker-4B is Apache 2.0, supports 100+ languages, 32k context, with strong published results</cite> — worth testing given your scientific/medical multilingual-adjacent corpus.
- Benchmarked on **BEIR, MTEB-R, MIRACL** — same benchmark family as embeddings, so you can reuse your MTEB-based shortlisting process for rerankers too.

---

## Summary Table — What to Benchmark Where

| Stage | Primary public benchmark | What you'll actually run |
|---|---|---|
| A4 Chunking | Vecta/FloTorch chunking benchmark (methodology reference) | Your own corpus, RAGAS context precision/recall across chunking variants |
| A5 Embedding | MTEB / MMTEB, BEIR | Shortlist 2-3 models, test recall@10/NDCG on your own golden set |
| A7 Storage | VectorDBBench, ANN-Benchmarks | Your own latency/recall test at your expected scale |
| A8 Evaluation | HotpotQA, MuSiQue, CRAG, RGB (sanity checks) | RAGAS + DeepEval + TruLens on your Claude-generated, human-reviewed golden set |
| Reranking | BEIR, MTEB-R, MIRACL | A/B test 2 rerankers on your golden set's eval lift |

Ready when you are — tell me how you want to proceed (pick models per stage, or want a decision-matrix format for narrowing down first).
