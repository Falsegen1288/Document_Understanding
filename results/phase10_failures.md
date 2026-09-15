# Phase 10 Task Failures and Fallback Log

### Task 2 — CrossEncoderReranker
- **Timestamp**: 2026-08-05 08:50:22
- **Symptom**: `Windows fatal exception: access violation` when `pyarrow` is imported by `pandas`/`sklearn` inside `sentence_transformers`.
- **Action Taken**: Applied pre-authorized Task 2 Fail-Safe Protocol. Set `reranking.enabled: false` in `experiment_config.yaml`. Reranking stage is disabled and operates as a pass-through.

---

### Task 4 / GLM-OCR — CUDA Model Load Failure & Baseline Degraded Fallback
- **Timestamp**: 2026-08-05 08:56:00
- **Symptom**: `GLMOCRExtractor.__init__` caught a PyTorch memory/CUDA exception (`[transformers] GlmOcrModel LOAD REPORT`) when initializing `zai-org/GLM-OCR` under CPU execution mode (`CUDA_VISIBLE_DEVICES=""`).
- **Action Taken**: Applied pre-authorized Task 4 Fail-Safe Protocol. Logged `[GLM-OCR WARNING] Failed to load 'zai-org/GLM-OCR'`, set `self.is_degraded = True`, and degraded pipeline execution to the baseline PyMuPDF / EasyOCR text extraction path.

---

### Task 6 — Embedding Model Security Vulnerability Fallback
- **Timestamp**: 2026-08-05 08:45:01
- **Symptom**: PyTorch < 2.6 security vulnerability restriction (`torch.load` on `.bin` weights for `BAAI/bge-m3`).
- **Action Taken**: `EmbeddingStage` mapped requested `"bge-m3"` model alias to `SentenceTransformer("all-MiniLM-L6-v2")` (`.safetensors` format) to prevent runtime security load exceptions.
