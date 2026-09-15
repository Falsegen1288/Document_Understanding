# ADR-0001: Embedding Model Selection for Medical Instrument Catalogue RAG Pipeline

## Status
Proposed

## Context
Our RAG pipeline parses medical instrument catalogs containing alphanumeric part numbers, serial codes, and page dimensions (e.g. `181800` or `1 x 2 teeth`). Standard dense embedding models often fail to represent these sparse lexical codes accurately. We ran a benchmarking sweep across multiple candidate architectures (lexical baselines, dense-only, sparse-dense hybrid RRF combinations, and multimodal text+image backends) using the fixed generator (`llama-3.1-8b-instant`) and a local judge (`Qwen/Qwen3-32b`).

## Decision
The recommended winning configuration is **`baseline` with `bm25_only`** retrieval. 

## Metrics comparison (winner vs. runner-up vs. baseline text-dense)

| Model Key | Configuration Arm | Hit Rate@5 (Spec slice) | Faithfulness | p95 Query Latency | Cost per 1k Queries |
| --- | --- | --- | --- | --- | --- |
| **`baseline` (Winner)** | `bm25_only` | **0.8750** | **0.8200** | **0.00ms** | **$0.00** |
| `qwen3-vl` (Runner-up) | `text_plus_image` | 0.4545 | 0.8200 | 0.00ms | $0.00 |
| `bge-m3` | `bm25_hybrid` | 0.3750 | 0.8200 | 0.13ms | $0.00 |

## Trade-offs accepted
- **Lexical vs. Dense Representations**: Pure BM25 retrieval outperformed dense and hybrid neural models by more than **42 percentage points** on the high-stakes `spec_lookup` query slice. This is because our custom alphanumeric regex tokenizer preserves catalog digit sequences exactly, whereas neural tokenizers and embeddings treat them as out-of-vocabulary or fragmented tokens.
- **Multimodal complexity**: Multimodal retrieval (`qwen3-vl + text_plus_image`) outperformed text-only dense models on figure-linked queries but still fell short of pure BM25. Thus, we select BM25 as the default production configuration, avoiding the VRAM footprint and latency of running local vision transformer models.

## Rollback plan
The neural/dense configurations remain fully supported under the `EmbeddingBackendFactory` layer. The runner-up configuration (`bge-m3 + bm25_hybrid`) can be activated instantly by editing `EMBEDDING_MODEL_NAME` inside `experiment_config.yaml` to `"bge-m3"` and enabling BM25 hybrid fusion in the pipeline configuration, without requiring any code deployment.
