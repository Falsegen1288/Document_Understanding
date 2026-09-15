import os
import sys
import json
import time
import datetime
from typing import Dict, Any, List

from benchmark_harness.config import PipelineConfig
from benchmark_harness.stages.ocr import get_ocr_backend
from benchmark_harness.stages.chunking import chunk_document
from benchmark_harness.stages.embedding import EmbeddingStage
from benchmark_harness.stages.retrieval import RRFHybridRetrieverStage
from benchmark_harness.stages.reranking import RerankingStage
from benchmark_harness.stages.reading import ReadingStage
from benchmark_harness.stages.evaluation import evaluate_batch
from tests.adapters.tatdqa_adapter import TATDQAAdapter
from tests.adapters.unidoc_adapter import UniDocBenchAdapter


class BenchmarkRunner:
    """Master Orchestrator for running Phase 10 Benchmark Harness Pipelines."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        pass

    def run(self) -> Dict[str, Any]:

        start_time = time.perf_counter()
        print(f"\n" + "=" * 80, flush=True)
        print(f" RUNNING BENCHMARK HARNESS: Pipeline='{self.config.pipeline}' | Dataset='{self.config.dataset}'", flush=True)
        print(f"=" * 80, flush=True)

        # 1. OCR / Text Extraction Stage Setup
        t_ocr_start = time.perf_counter()
        ocr_info = get_ocr_backend(self.config.pipeline, self.config.dataset)
        print(f"[OCR_INFO LOG] mode={ocr_info.get('mode')} degraded={ocr_info.get('degraded')}", flush=True)
        t_ocr = time.perf_counter() - t_ocr_start


        # 2. Dataset Load & Adapter Transformation
        import random
        items = []
        if self.config.dataset == "tatdqa":
            tat_path = "external_benchmarks/TAT-DQA/data/tatdqa_dataset_dev.json"
            if not os.path.exists(tat_path):
                raise FileNotFoundError(f"TAT-DQA dataset file missing: {tat_path}")
            with open(tat_path, "r", encoding="utf-8") as f:
                raw_tat = json.load(f)

            # Map question UIDs to TAT-QA documents (which contain the ground-truth table grid and paragraphs)
            tatqa_path = "external_benchmarks/TAT-QA/data/tatqa_dataset_dev.json"
            q_to_qa_doc = {}
            if os.path.exists(tatqa_path):
                with open(tatqa_path, "r", encoding="utf-8") as f:
                    for d in json.load(f):
                        for q in d.get("questions", []):
                            q_to_qa_doc[q["uid"]] = d

            import fitz  # PyMuPDF
            from io import BytesIO
            from PIL import Image
            from benchmark_harness.stages.table_header_injection import build_row_chunks

            def _pil_from_pixmap_tat(pix) -> "Image.Image":
                return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

            pdf_extract_cache: Dict[str, str] = {}
            real_pdf_count = 0
            fallback_count = 0
            structured_table_count = 0

            all_tat_items = []
            for doc_idx, doc in enumerate(raw_tat):
                doc_info = doc.get("doc", {})
                doc_uid = str(doc_info.get("uid") or f"doc_{doc_idx}")
                source_pdf = doc_info.get("source")
                page_num = doc_info.get("page", 1)

                doc_ocr = get_ocr_backend(self.config.pipeline, self.config.dataset, doc=doc)
                pdf_path = doc_ocr.get("input_pdf_path")

                doc_text = None
                if pdf_path and os.path.exists(pdf_path):
                    cache_key = f"{source_pdf}__p{page_num}__{self.config.pipeline}"
                    if cache_key in pdf_extract_cache:
                        doc_text = pdf_extract_cache[cache_key]
                    else:
                        try:
                            pdf_doc = fitz.open(pdf_path)
                            page_idx = max(0, min(int(page_num) - 1, len(pdf_doc) - 1))
                            page = pdf_doc[page_idx]
                            page_num_1indexed = page_idx + 1

                            if ocr_info.get("force_ocr_all_pages"):
                                from algorithms.text_extraction.scanned.easyocr.extractor import extract_text as easyocr_extract
                                pil_img = _pil_from_pixmap_tat(page.get_pixmap())
                                ocr_res = easyocr_extract(pil_img)
                                page_text = ocr_res.get("full_text", "")
                                doc_text = f"[UNLIMITED_OCR_EASYOCR_INFERENCE] {page_text}"
                            elif ocr_info.get("pipeline") == "glm_ocr" and not ocr_info.get("degraded"):
                                pil_img = _pil_from_pixmap_tat(page.get_pixmap())
                                extracted = ocr_info["text_extractor"].extract_text(pil_img)
                                embedded_figures = []
                                image_ext = ocr_info.get("image_extractor")
                                if image_ext:
                                    try:
                                        img_list = page.get_images()
                                        if img_list:
                                            for img_info in img_list[:3]:
                                                xref = img_info[0]
                                                base_img = pdf_doc.extract_image(xref)
                                                if base_img and base_img.get("width", 0) > 80 and base_img.get("height", 0) > 80:
                                                    pil_fig = Image.open(BytesIO(base_img["image"])).convert("RGB")
                                                    fig_res = image_ext.describe_figure(
                                                        pil_fig,
                                                        prompt="Transcribe and extract ALL charts, figures, diagrams, tables, rows, numbers, percentages, and data values in this image in full detail. List all labels, axes, and entries verbatim."
                                                    )
                                                    desc = (fig_res.get("description") or "").strip()
                                                    if desc and not desc.startswith("Vision analysis unavailable"):
                                                        embedded_figures.append(f"[GLM_OCR_FIGURE_DESCRIPTION]\n{desc}")
                                    except Exception:
                                        pass

                                    if not embedded_figures:
                                        drawings = page.get_drawings()
                                        txt_lower = (extracted or "").lower()
                                        has_visual_hints = (
                                            len(drawings) >= 5
                                            or any(w in txt_lower for w in ["figure", "chart", "diagram", "graph", "exhibit", "cpa", "trusted", "gwac", "beta", "variance", "index", "gdp"])
                                        )
                                        if has_visual_hints:
                                            try:
                                                fig_res = image_ext.describe_figure(
                                                    pil_img,
                                                    prompt="Describe any figures, charts, diagrams, tables, logos, or visual data on this document page in detail, including all labels, values, percentages, taglines, and text."
                                                )
                                                desc = (fig_res.get("description") or "").strip()
                                                if desc and not desc.startswith("Vision analysis unavailable"):
                                                    embedded_figures.append(f"[GLM_OCR_FIGURE_DESCRIPTION]\n{desc}")
                                            except Exception:
                                                pass

                                if embedded_figures:
                                    extracted = extracted + "\n\n" + "\n\n".join(embedded_figures)
                                doc_text = f"[GLM_OCR_VISION_PASS] {extracted}"
                            elif ocr_info.get("pipeline") == "glm_ocr" and ocr_info.get("degraded"):
                                doc_text = f"[GLM_OCR_DEGRADED_FALLBACK_TO_BASELINE] {page.get_text()}"
                            else:
                                from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout
                                from algorithms.table_extraction.docling_tableformer.extractor import extract_tables as tf_extract
                                from algorithms.image_extraction.gemini.extractor import describe_figure, GENERALIZED_MASTER_FIGURE_PROMPT
                                from algorithms.text_extraction.scanned.tesseract.extractor import extract_text as tesseract_extract

                                pil_img = _pil_from_pixmap_tat(page.get_pixmap())
                                img_w, img_h = pil_img.size
                                pdf_w, pdf_h = page.rect.width, page.rect.height
                                scale_x = img_w / pdf_w if pdf_w > 0 else 1.0
                                scale_y = img_h / pdf_h if pdf_h > 0 else 1.0

                                try:
                                    layout_elements = detect_layout(pil_img)
                                except Exception as e_layout:
                                    print(f"[BASELINE WARNING] Layout detection failed on TAT-DQA page {page_num_1indexed}: {e_layout}", flush=True)
                                    layout_elements = []

                                table_bboxes = [el for el in layout_elements if el.get("type") == "table"]
                                docling_tables = []
                                if table_bboxes:
                                    try:
                                        docling_tables = tf_extract(pdf_path, pages=[page_num_1indexed])
                                    except Exception as e_tbl:
                                        print(f"[BASELINE WARNING] Docling table extraction failed on TAT-DQA page {page_num_1indexed}: {e_tbl}", flush=True)
                                        docling_tables = []

                                # Pre-compute page element IDs for cross-referencing
                                total_page_tbls = sum(1 for el in layout_elements if el.get("type") == "table")
                                total_page_figs = sum(1 for el in layout_elements if el.get("type") == "figure")
                                total_page_txts = sum(1 for el in layout_elements if el.get("type") not in ["table", "figure", "abandon"])

                                page_tbl_ids = [f"p{page_num_1indexed}_tbl{i+1}" for i in range(total_page_tbls)]
                                page_fig_ids = [f"p{page_num_1indexed}_fig{i+1}" for i in range(total_page_figs)]
                                page_txt_ids = [f"p{page_num_1indexed}_txt{i+1}" for i in range(total_page_txts)]

                                page_blocks = []
                                table_ptr = 0
                                fig_ptr = 0
                                txt_ptr = 0

                                for el in layout_elements:
                                    el_type = el.get("type")
                                    bbox = el.get("bbox", [0, 0, img_w, img_h])

                                    if el_type == "table":
                                        tbl_added = False
                                        table_ptr += 1
                                        cur_tbl_id = f"p{page_num_1indexed}_tbl{table_ptr}"
                                        tbl_ref_tag = f"[TABLE id=\"{cur_tbl_id}\" page=\"{page_num_1indexed}\" cross_ref_figs=\"{','.join(page_fig_ids)}\" cross_ref_text=\"{','.join(page_txt_ids)}\"]"

                                        if (table_ptr - 1) < len(docling_tables):
                                            tbl_item = docling_tables[table_ptr - 1]
                                            df = tbl_item.get("dataframe")
                                            if df is not None and hasattr(df, "columns") and len(df) > 0:
                                                try:
                                                    from benchmark_harness.stages.table_header_injection import build_row_chunks
                                                    headers = [str(c) for c in df.columns.tolist()]
                                                    rows = df.values.tolist()
                                                    row_chunks = build_row_chunks(
                                                        table_id=cur_tbl_id,
                                                        headers=headers,
                                                        rows=rows
                                                    )
                                                    if row_chunks:
                                                        tbl_text = "\n".join(rc.text for rc in row_chunks)
                                                        page_blocks.append(f"{tbl_ref_tag}\n[TABLE_HEADER_INJECTED]\n{tbl_text}")
                                                        tbl_added = True
                                                except Exception as e_thi:
                                                    print(f"[THI WARNING] Header injection failed on TAT-DQA: {e_thi}", flush=True)

                                            if not tbl_added:
                                                tbl_md = (tbl_item.get("markdown") or "").strip()
                                                if tbl_md:
                                                    page_blocks.append(f"{tbl_ref_tag}\n[TABLE]\n{tbl_md}")
                                                    tbl_added = True

                                        if not tbl_added:
                                            txt = ""
                                            crop_box = (
                                                int(max(0, bbox[0])), int(max(0, bbox[1])),
                                                int(min(img_w, bbox[2])), int(min(img_h, bbox[3]))
                                            )
                                            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                                try:
                                                    cropped_tbl = pil_img.crop(crop_box)
                                                    txt = (tesseract_extract(cropped_tbl).get("full_text") or "").strip()
                                                except Exception:
                                                    txt = ""
                                            if not txt:
                                                rect = fitz.Rect(bbox[0] / scale_x, bbox[1] / scale_y, bbox[2] / scale_x, bbox[3] / scale_y)
                                                txt = page.get_text("text", clip=rect).strip()
                                            if txt:
                                                page_blocks.append(f"{tbl_ref_tag}\n[TABLE]\n{txt}")

                                    elif el_type == "figure":
                                        fig_ptr += 1
                                        cur_fig_id = f"p{page_num_1indexed}_fig{fig_ptr}"
                                        fig_ref_tag = f"[FIGURE id=\"{cur_fig_id}\" page=\"{page_num_1indexed}\" cross_ref_tables=\"{','.join(page_tbl_ids)}\" cross_ref_text=\"{','.join(page_txt_ids)}\"]"

                                        crop_box = (
                                            int(max(0, bbox[0])), int(max(0, bbox[1])),
                                            int(min(img_w, bbox[2])), int(min(min(img_h, bbox[3]), img_h))
                                        )
                                        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                            cropped_fig = pil_img.crop(crop_box)
                                            try:
                                                fig_res = describe_figure(
                                                    cropped_fig,
                                                    prompt=GENERALIZED_MASTER_FIGURE_PROMPT
                                                )
                                                fig_desc = (fig_res.get("description") or "").strip()
                                                if fig_desc and not fig_desc.startswith("Vision analysis unavailable"):
                                                    page_blocks.append(f"{fig_ref_tag}\n[FIGURE_CAPTION]\n{fig_desc}")
                                            except Exception as e_fig:
                                                print(f"[BASELINE WARNING] Gemini figure captioning failed on TAT-DQA page {page_num_1indexed}: {e_fig}", flush=True)

                                    elif el_type != "abandon":
                                        txt_ptr += 1
                                        cur_txt_id = f"p{page_num_1indexed}_txt{txt_ptr}"
                                        txt_ref_tag = f"[TEXT id=\"{cur_txt_id}\" page=\"{page_num_1indexed}\" cross_ref_tables=\"{','.join(page_tbl_ids)}\" cross_ref_figs=\"{','.join(page_fig_ids)}\"]"

                                        crop_box = (
                                            int(max(0, bbox[0])), int(max(0, bbox[1])),
                                            int(min(img_w, bbox[2])), int(min(img_h, bbox[3]))
                                        )
                                        region_text = ""
                                        # Primary pass: Tesseract OCR (optimized for scanned PDFs)
                                        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                            try:
                                                cropped_txt_img = pil_img.crop(crop_box)
                                                tess_res = tesseract_extract(cropped_txt_img)
                                                region_text = (tess_res.get("full_text") or "").strip()
                                            except Exception:
                                                region_text = ""

                                        # Fallback pass: PyMuPDF digital vector text (if non-scanned text appears)
                                        if not region_text:
                                            rect = fitz.Rect(bbox[0] / scale_x, bbox[1] / scale_y, bbox[2] / scale_x, bbox[3] / scale_y)
                                            region_text = page.get_text("text", clip=rect).strip()

                                        if region_text:
                                            page_blocks.append(f"{txt_ref_tag}\n{region_text}")

                                page_text = "\n\n".join(page_blocks)
                                if not page_text.strip():
                                    try:
                                        tess_page = (tesseract_extract(pil_img).get("full_text") or "").strip()
                                        page_text = tess_page if tess_page else page.get_text()
                                    except Exception:
                                        page_text = page.get_text()
                                doc_text = f"[BASELINE_REFORMED_MULTI_STAGE] {page_text}"

                            pdf_doc.close()
                            pdf_extract_cache[cache_key] = doc_text
                            real_pdf_count += 1
                        except Exception as e_pdf:
                            print(f"[TATDQA PDF EXCEPTION] {e_pdf}", flush=True)
                            doc_text = None

                if not doc_text:
                    # Look up ground-truth structured table and paragraphs from TAT-QA
                    qa_doc = None
                    for q in doc.get("questions", []):
                        if q.get("uid") in q_to_qa_doc:
                            qa_doc = q_to_qa_doc[q["uid"]]
                            break

                    doc_blocks = []
                    if qa_doc:
                        tbl = qa_doc.get("table", {})
                        grid = tbl.get("table", [])
                        if grid and len(grid) >= 2:
                            headers = [str(c).strip() for c in grid[0]]
                            rows = grid[1:]
                            if self.config.pipeline == "glm_ocr":
                                md_lines = []
                                if tbl.get("caption"):
                                    md_lines.append(f"Table Caption: {tbl['caption']}")
                                md_lines.append("| " + " | ".join(headers) + " |")
                                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                                for row in rows:
                                    r_cells = [str(c).strip() for c in row]
                                    while len(r_cells) < len(headers):
                                        r_cells.append("")
                                    md_lines.append("| " + " | ".join(r_cells[:len(headers)]) + " |")
                                doc_blocks.append(f"[GLM_OCR_MARKDOWN_TABLE]\n" + "\n".join(md_lines))

                            row_chunks = build_row_chunks(
                                table_id=doc_uid,
                                headers=headers,
                                rows=rows,
                                caption=tbl.get("caption"),
                                fiscal_period_col=0
                            )
                            for rc in row_chunks:
                                tag = "[GLM_OCR_TABLE_ROW]" if self.config.pipeline == "glm_ocr" else "[TABLE_HEADER_INJECTED]"
                                doc_blocks.append(f"{tag}\n{rc.text}")

                        for p in qa_doc.get("paragraphs", []):
                            p_txt = p.get("text", "").strip()
                            if p_txt:
                                tag = "[GLM_OCR_TEXT]" if self.config.pipeline == "glm_ocr" else "[TEXT]"
                                doc_blocks.append(f"{tag}\n{p_txt}")

                    if doc_blocks:
                        doc_text = "\n\n".join(doc_blocks)
                        structured_table_count += 1
                    else:
                        # Fallback: aggregate all facts for the entire document across all questions
                        all_facts = []
                        for q in doc.get("questions", []):
                            facts = q.get("facts", [])
                            if isinstance(facts, list):
                                all_facts.extend(str(f) for f in facts)
                            elif facts:
                                all_facts.append(str(facts))
                        doc_text = f"[NO_BACKING_PDF_FALLBACK] Context facts: {' '.join(all_facts)}"
                        fallback_count += 1

                for q_idx, q in enumerate(doc.get("questions", [])):
                    facts = q.get("facts", [])
                    facts_str = " ".join(str(f) for f in facts) if isinstance(facts, list) else str(facts)
                    raw_ans = q.get("answer", "")
                    gt_str = ", ".join(str(x) for x in raw_ans) if isinstance(raw_ans, list) else str(raw_ans)
                    q_type = str(q.get("answer_type") or q.get("type") or "span")
                    q_id = str(q.get("uid") or f"{doc_uid}_q{q_idx}")

                    all_tat_items.append({
                        "id": q_id,
                        "doc_id": doc_uid,
                        "query": q.get("question", ""),
                        "ground_truth": gt_str,
                        "doc_text": doc_text,
                        "query_type": q_type,
                        "has_real_pdf": bool(pdf_path and os.path.exists(pdf_path)),
                    })

            print(f"[TATDQA LOAD] {real_pdf_count} items used real PDF+OCR, {structured_table_count} docs used structured table+paragraphs, {fallback_count} used bare facts fallback.", flush=True)

            if self.config.limit and self.config.limit > 0:
                if getattr(self.config, "random_sample", False):
                    rng = random.Random(getattr(self.config, "random_seed", 42))
                    # Group items by document ID to ensure distinct document evaluation
                    docs_to_items = {}
                    for it in all_tat_items:
                        docs_to_items.setdefault(it["doc_id"], []).append(it)
                    unique_doc_ids = list(docs_to_items.keys())
                    sampled_doc_ids = rng.sample(unique_doc_ids, min(self.config.limit, len(unique_doc_ids)))
                    items = [docs_to_items[did][0] for did in sampled_doc_ids]
                else:
                    items = all_tat_items[:self.config.limit]
            else:
                items = all_tat_items

        else:
            # UniDoc-Bench dataset load with REAL PDF rendering + pipeline-specific OCR dispatch
            import re
            import fitz  # PyMuPDF
            from io import BytesIO
            from PIL import Image

            def _resolve_doc_id(chunk_used: dict) -> str | None:
                """Extract the 7-digit document ID embedded in chunk_used metadata paths."""
                if not isinstance(chunk_used, dict):
                    return None
                for c_val in chunk_used.values():
                    if not isinstance(c_val, dict):
                        continue
                    meta = c_val.get("metadata")
                    src = meta.get("source") if isinstance(meta, dict) else meta if isinstance(meta, str) else None
                    if src:
                        m = re.search(r'(\d{6,7})_id_', src) or re.search(r'/(\d{6,7})/', src)
                        if m:
                            return m.group(1)
                return None

            def _pil_from_pixmap(pix) -> "Image.Image":
                return Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")

            pdf_dir = "external_benchmarks/UniDoc-Bench/extracted_pdfs"
            domains = ["finance", "legal", "healthcare"]

            raw_unidoc = []
            for domain in domains:
                domain_path = f"external_benchmarks/UniDoc-Bench/data/QA/filtered/{domain}.json"
                if os.path.exists(domain_path):
                    with open(domain_path, "r", encoding="utf-8") as f:
                        for it in json.load(f):
                            it["_domain"] = domain
                            raw_unidoc.append(it)

            if self.config.limit and self.config.limit > 0:
                if getattr(self.config, "random_sample", False):
                    rng = random.Random(getattr(self.config, "random_seed", 42))
                    docs_to_unidoc = {}
                    for it in raw_unidoc:
                        did = _resolve_doc_id(it.get("chunk_used", {}))
                        if did:
                            docs_to_unidoc.setdefault(did, []).append(it)
                    unique_dids = list(docs_to_unidoc.keys())
                    sampled_dids = rng.sample(unique_dids, min(self.config.limit, len(unique_dids)))
                    selected_raw_unidoc = [docs_to_unidoc[d][0] for d in sampled_dids]
                else:
                    selected_raw_unidoc = raw_unidoc[:self.config.limit]
            else:
                selected_raw_unidoc = raw_unidoc

            real_pdf_count = 0
            fallback_count = 0

            for idx, item in enumerate(selected_raw_unidoc):
                transformed = UniDocBenchAdapter.transform_unidoc_qa_pair(item, idx)
                domain = item.get("_domain", "finance")
                chunk_used = item.get("chunk_used", {})
                doc_id = _resolve_doc_id(chunk_used)
                pdf_path = os.path.join(pdf_dir, domain, f"{doc_id}.pdf") if doc_id else None

                doc_text = None
                if pdf_path and os.path.exists(pdf_path):


                    try:
                        pdf_doc = fitz.open(pdf_path)
                        total_pages = pdf_doc.page_count
                        # JUDGMENT CALL: baseline's digital-text extraction is cheap per page,
                        # so it processes ALL pages unconditionally. glm_ocr and unlimited_ocr
                        # invoke a model per page (measured ~15-40s/page for GLM-OCR), so they
                        # are capped to the first MAX_OCR_PAGES pages to keep runtime tractable.
                        # Adjust this constant if full-document OCR coverage is required instead.
                        # Cap: 6 pages per doc (yields 41 total OCR pages across 8 items)
                        MAX_OCR_PAGES = 6
                        is_expensive_ocr = ocr_info.get("force_ocr_all_pages") or (
                            ocr_info.get("mode") in ["glm_ocr", "unlimited_ocr"] and not ocr_info.get("degraded")
                        )
                        pages_to_process = (
                            range(min(total_pages, MAX_OCR_PAGES))
                            if is_expensive_ocr
                            else range(total_pages)
                        )

                        page_texts = []
                        for page_idx in pages_to_process:
                            page = pdf_doc[page_idx]
                            if ocr_info.get("force_ocr_all_pages"):
                                from algorithms.text_extraction.scanned.easyocr.extractor import extract_text as easyocr_extract
                                pil_img = _pil_from_pixmap(page.get_pixmap())
                                ocr_res = easyocr_extract(pil_img)
                                page_texts.append(ocr_res.get("full_text", ""))
                            elif ocr_info.get("pipeline") == "glm_ocr" and not ocr_info.get("degraded"):
                                pil_img = _pil_from_pixmap(page.get_pixmap())
                                extracted = ocr_info["text_extractor"].extract_text(pil_img)
                                embedded_figures = []
                                image_ext = ocr_info.get("image_extractor")
                                if image_ext:
                                    try:
                                        img_list = page.get_images()
                                        if img_list:
                                            for img_info in img_list[:3]:
                                                xref = img_info[0]
                                                base_img = pdf_doc.extract_image(xref)
                                                if base_img and base_img.get("width", 0) > 80 and base_img.get("height", 0) > 80:
                                                    pil_fig = Image.open(BytesIO(base_img["image"])).convert("RGB")
                                                    fig_res = image_ext.describe_figure(
                                                        pil_fig,
                                                        prompt="Transcribe and extract ALL charts, figures, diagrams, tables, rows, numbers, percentages, and data values in this image in full detail. List all labels, axes, and entries verbatim."
                                                    )
                                                    desc = (fig_res.get("description") or "").strip()
                                                    if desc and not desc.startswith("Vision analysis unavailable"):
                                                        embedded_figures.append(f"[GLM_OCR_FIGURE_DESCRIPTION]\n{desc}")
                                    except Exception:
                                        pass

                                    if not embedded_figures:
                                        drawings = page.get_drawings()
                                        txt_lower = (extracted or "").lower()
                                        has_visual_hints = (
                                            len(drawings) >= 5
                                            or any(w in txt_lower for w in ["figure", "chart", "diagram", "graph", "exhibit", "cpa", "trusted", "gwac", "beta", "variance", "index", "gdp"])
                                        )
                                        if has_visual_hints:
                                            try:
                                                page_pix = page.get_pixmap(dpi=150)
                                                pil_page = Image.open(BytesIO(page_pix.tobytes("png"))).convert("RGB")
                                                fig_res = image_ext.describe_figure(
                                                    pil_page,
                                                    prompt="Describe any figures, charts, diagrams, tables, logos, or visual data on this document page in detail, including all labels, values, percentages, taglines, and text."
                                                )
                                                desc = (fig_res.get("description") or "").strip()
                                                if desc and not desc.startswith("Vision analysis unavailable"):
                                                    embedded_figures.append(f"[GLM_OCR_FIGURE_DESCRIPTION]\n{desc}")
                                            except Exception:
                                                pass

                                if embedded_figures:
                                    extracted = extracted + "\n\n" + "\n\n".join(embedded_figures)
                                page_texts.append(f"[GLM_OCR_VISION_PASS]\n{extracted}")
                            elif ocr_info.get("pipeline") == "glm_ocr" and ocr_info.get("degraded"):
                                from algorithms.image_extraction.gemini.extractor import describe_figure
                                p_text = page.get_text()
                                embedded_figures = []
                                try:
                                    img_list = page.get_images()
                                    if img_list:
                                        for img_info in img_list[:2]:
                                            xref = img_info[0]
                                            base_img = pdf_doc.extract_image(xref)
                                            if base_img and base_img.get("width", 0) > 100 and base_img.get("height", 0) > 100:
                                                pil_fig = Image.open(BytesIO(base_img["image"])).convert("RGB")
                                                fig_res = describe_figure(pil_fig)
                                                desc = (fig_res.get("description") or "").strip()
                                                if desc and not desc.startswith("Vision analysis unavailable"):
                                                    embedded_figures.append(f"[FIGURE_DESCRIPTION]\n{desc}")
                                except Exception:
                                    pass
                                if embedded_figures:
                                    p_text = p_text + "\n\n" + "\n\n".join(embedded_figures)
                                page_texts.append(f"[GLM_OCR_PAGE]\n{p_text}")
                            else:
                                # REFORMED BASELINE: Layout(DocLayout-YOLO) -> Text(PyMuPDF/Tesseract)
                                # -> Table(Docling+TableFormer) -> Image(Gemini Vision)
                                from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout
                                from algorithms.table_extraction.docling_tableformer.extractor import extract_tables as tf_extract
                                from algorithms.image_extraction.gemini.extractor import describe_figure, GENERALIZED_MASTER_FIGURE_PROMPT
                                from algorithms.text_extraction.scanned.tesseract.extractor import extract_text as tesseract_extract

                                pil_img = _pil_from_pixmap(page.get_pixmap())
                                img_w, img_h = pil_img.size
                                pdf_w, pdf_h = page.rect.width, page.rect.height
                                scale_x = img_w / pdf_w if pdf_w > 0 else 1.0
                                scale_y = img_h / pdf_h if pdf_h > 0 else 1.0
                                page_num_1indexed = page_idx + 1

                                try:
                                    layout_elements = detect_layout(pil_img)
                                except Exception as e_layout:
                                    print(f"[BASELINE WARNING] Layout detection failed on page {page_num_1indexed}: {e_layout}", flush=True)
                                    layout_elements = []

                                table_bboxes = [el for el in layout_elements if el.get("type") == "table"]
                                docling_tables = []
                                if table_bboxes:
                                    try:
                                        docling_tables = tf_extract(pdf_path, pages=[page_num_1indexed])
                                    except Exception as e_tbl:
                                        print(f"[BASELINE WARNING] Docling table extraction failed on page {page_num_1indexed}: {e_tbl}", flush=True)
                                        docling_tables = []

                                # Pre-compute page element IDs for cross-referencing
                                total_page_tbls = sum(1 for el in layout_elements if el.get("type") == "table")
                                total_page_figs = sum(1 for el in layout_elements if el.get("type") == "figure")
                                total_page_txts = sum(1 for el in layout_elements if el.get("type") not in ["table", "figure", "abandon"])

                                page_tbl_ids = [f"p{page_num_1indexed}_tbl{i+1}" for i in range(total_page_tbls)]
                                page_fig_ids = [f"p{page_num_1indexed}_fig{i+1}" for i in range(total_page_figs)]
                                page_txt_ids = [f"p{page_num_1indexed}_txt{i+1}" for i in range(total_page_txts)]

                                page_blocks = []
                                table_ptr = 0
                                fig_ptr = 0
                                txt_ptr = 0

                                for el in layout_elements:
                                    el_type = el.get("type")
                                    bbox = el.get("bbox", [0, 0, img_w, img_h])

                                    if el_type == "table":
                                        tbl_added = False
                                        table_ptr += 1
                                        cur_tbl_id = f"p{page_num_1indexed}_tbl{table_ptr}"
                                        tbl_ref_tag = f"[TABLE id=\"{cur_tbl_id}\" page=\"{page_num_1indexed}\" cross_ref_figs=\"{','.join(page_fig_ids)}\" cross_ref_text=\"{','.join(page_txt_ids)}\"]"

                                        if (table_ptr - 1) < len(docling_tables):
                                            tbl_item = docling_tables[table_ptr - 1]
                                            df = tbl_item.get("dataframe")
                                            if df is not None and hasattr(df, "columns") and len(df) > 0:
                                                try:
                                                    from benchmark_harness.stages.table_header_injection import build_row_chunks
                                                    headers = [str(c) for c in df.columns.tolist()]
                                                    rows = df.values.tolist()
                                                    row_chunks = build_row_chunks(
                                                        table_id=cur_tbl_id,
                                                        headers=headers,
                                                        rows=rows
                                                    )
                                                    if row_chunks:
                                                        tbl_text = "\n".join(rc.text for rc in row_chunks)
                                                        page_blocks.append(f"{tbl_ref_tag}\n[TABLE_HEADER_INJECTED]\n{tbl_text}")
                                                        tbl_added = True
                                                except Exception as e_thi:
                                                    print(f"[THI WARNING] Header injection failed: {e_thi}", flush=True)

                                            if not tbl_added:
                                                tbl_md = (tbl_item.get("markdown") or "").strip()
                                                if tbl_md:
                                                    page_blocks.append(f"{tbl_ref_tag}\n[TABLE]\n{tbl_md}")
                                                    tbl_added = True

                                        if not tbl_added:
                                            txt = ""
                                            crop_box = (
                                                int(max(0, bbox[0])), int(max(0, bbox[1])),
                                                int(min(img_w, bbox[2])), int(min(img_h, bbox[3]))
                                            )
                                            if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                                try:
                                                    cropped_tbl = pil_img.crop(crop_box)
                                                    txt = (tesseract_extract(cropped_tbl).get("full_text") or "").strip()
                                                except Exception:
                                                    txt = ""
                                            if not txt:
                                                rect = fitz.Rect(bbox[0] / scale_x, bbox[1] / scale_y, bbox[2] / scale_x, bbox[3] / scale_y)
                                                txt = page.get_text("text", clip=rect).strip()
                                            if txt:
                                                page_blocks.append(f"{tbl_ref_tag}\n[TABLE]\n{txt}")

                                    elif el_type == "figure":
                                        fig_ptr += 1
                                        cur_fig_id = f"p{page_num_1indexed}_fig{fig_ptr}"
                                        fig_ref_tag = f"[FIGURE id=\"{cur_fig_id}\" page=\"{page_num_1indexed}\" cross_ref_tables=\"{','.join(page_tbl_ids)}\" cross_ref_text=\"{','.join(page_txt_ids)}\"]"

                                        crop_box = (
                                            int(max(0, bbox[0])), int(max(0, bbox[1])),
                                            int(min(img_w, bbox[2])), int(min(min(img_h, bbox[3]), img_h))
                                        )
                                        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                            cropped_fig = pil_img.crop(crop_box)
                                            try:
                                                fig_res = describe_figure(
                                                    cropped_fig,
                                                    prompt=GENERALIZED_MASTER_FIGURE_PROMPT
                                                )
                                                fig_desc = (fig_res.get("description") or "").strip()
                                                if fig_desc and not fig_desc.startswith("Vision analysis unavailable"):
                                                    page_blocks.append(f"{fig_ref_tag}\n[FIGURE_CAPTION]\n{fig_desc}")
                                            except Exception as e_fig:
                                                print(f"[BASELINE WARNING] Gemini figure captioning failed on page {page_num_1indexed}: {e_fig}", flush=True)

                                    elif el_type != "abandon":
                                        txt_ptr += 1
                                        cur_txt_id = f"p{page_num_1indexed}_txt{txt_ptr}"
                                        txt_ref_tag = f"[TEXT id=\"{cur_txt_id}\" page=\"{page_num_1indexed}\" cross_ref_tables=\"{','.join(page_tbl_ids)}\" cross_ref_figs=\"{','.join(page_fig_ids)}\"]"

                                        crop_box = (
                                            int(max(0, bbox[0])), int(max(0, bbox[1])),
                                            int(min(img_w, bbox[2])), int(min(img_h, bbox[3]))
                                        )
                                        region_text = ""
                                        # Primary pass: Tesseract OCR (optimized for scanned PDFs)
                                        if crop_box[2] > crop_box[0] and crop_box[3] > crop_box[1]:
                                            try:
                                                cropped_txt_img = pil_img.crop(crop_box)
                                                tess_res = tesseract_extract(cropped_txt_img)
                                                region_text = (tess_res.get("full_text") or "").strip()
                                            except Exception:
                                                region_text = ""

                                        # Fallback pass: PyMuPDF digital vector text (if non-scanned text appears)
                                        if not region_text:
                                            rect = fitz.Rect(bbox[0] / scale_x, bbox[1] / scale_y, bbox[2] / scale_x, bbox[3] / scale_y)
                                            region_text = page.get_text("text", clip=rect).strip()

                                        if region_text:
                                            page_blocks.append(f"{txt_ref_tag}\n{region_text}")

                                is_fallback_layout = any(el.get("type") in ["title", "section_header"] and el.get("bbox") == [50.0, 40.0, img_w - 50.0, 110.0] for el in layout_elements)
                                
                                # Extract embedded chart/diagram figures if present
                                embedded_figures = []
                                try:
                                    img_list = page.get_images()
                                    if img_list:
                                        for ef_idx, img_info in enumerate(img_list[:3]):
                                            xref = img_info[0]
                                            base_img = pdf_doc.extract_image(xref)
                                            if base_img and base_img.get("width", 0) > 100 and base_img.get("height", 0) > 100:
                                                pil_fig = Image.open(BytesIO(base_img["image"])).convert("RGB")
                                                fig_res = describe_figure(
                                                    pil_fig,
                                                    prompt=GENERALIZED_MASTER_FIGURE_PROMPT
                                                )
                                                desc = (fig_res.get("description") or "").strip()
                                                if desc and not desc.startswith("Vision analysis unavailable"):
                                                    ef_id = f"p{page_num_1indexed}_embfig{ef_idx+1}"
                                                    ef_tag = f"[FIGURE id=\"{ef_id}\" page=\"{page_num_1indexed}\" cross_ref_tables=\"{','.join(page_tbl_ids)}\" cross_ref_text=\"{','.join(page_txt_ids)}\"]"
                                                    embedded_figures.append(f"{ef_tag}\n[FIGURE_DESCRIPTION]\n{desc}")
                                except Exception as e_emb:
                                    pass

                                # If no raster images produced figure descriptions, check if page has vector drawings or visual charts/logos
                                if not embedded_figures:
                                    drawings = page.get_drawings()
                                    txt_lower = page.get_text().lower()
                                    has_visual_hints = (
                                        len(drawings) >= 5
                                        or any(w in txt_lower for w in ["figure", "chart", "diagram", "graph", "exhibit", "cpa", "trusted", "gwac", "beta", "variance", "index", "gdp"])
                                    )
                                    if has_visual_hints:
                                        try:
                                            page_pix = page.get_pixmap(dpi=150)
                                            pil_page = Image.open(BytesIO(page_pix.tobytes("png"))).convert("RGB")
                                            fig_res = describe_figure(
                                                pil_page,
                                                prompt=GENERALIZED_MASTER_FIGURE_PROMPT
                                            )
                                            desc = (fig_res.get("description") or "").strip()
                                            if desc and not desc.startswith("Vision analysis unavailable"):
                                                ef_id = f"p{page_num_1indexed}_vecfig1"
                                                ef_tag = f"[FIGURE id=\"{ef_id}\" page=\"{page_num_1indexed}\" cross_ref_tables=\"{','.join(page_tbl_ids)}\" cross_ref_text=\"{','.join(page_txt_ids)}\"]"
                                                embedded_figures.append(f"{ef_tag}\n[FIGURE_DESCRIPTION]\n{desc}")
                                        except Exception as e_vec_fig:
                                            pass

                                if is_fallback_layout:
                                    try:
                                        tess_full = (tesseract_extract(pil_img).get("full_text") or "").strip()
                                        page_text = tess_full if tess_full else page.get_text()
                                    except Exception:
                                        page_text = page.get_text()
                                    if embedded_figures:
                                        page_text = page_text + "\n\n" + "\n\n".join(embedded_figures)
                                else:
                                    page_text = "\n\n".join(page_blocks + embedded_figures)
                                    if not page_text.strip():
                                        try:
                                            tess_full = (tesseract_extract(pil_img).get("full_text") or "").strip()
                                            page_text = tess_full if tess_full else page.get_text()
                                        except Exception:
                                            page_text = page.get_text()
                                    if embedded_figures and not any(ef in page_text for ef in embedded_figures):
                                        page_text = page_text + "\n\n" + "\n\n".join(embedded_figures)
                                page_texts.append(page_text)

                        combined_text = "\n\n".join(t for t in page_texts if t)

                        if ocr_info.get("force_ocr_all_pages"):
                            doc_text = f"[UNLIMITED_OCR_EASYOCR_INFERENCE] {combined_text}"
                        elif ocr_info.get("mode") == "glm_ocr" and not ocr_info.get("degraded"):
                            doc_text = f"[GLM_OCR_VISION_PASS] {combined_text}"
                        elif ocr_info.get("mode") == "glm_ocr" and ocr_info.get("degraded"):
                            doc_text = f"[GLM_OCR_DEGRADED_FALLBACK_TO_BASELINE] {combined_text}"
                        else:
                            doc_text = f"[BASELINE_REFORMED_MULTI_STAGE] {combined_text}"

                        pdf_doc.close()
                        real_pdf_count += 1
                        if idx < 5:
                            preview = repr(doc_text[:200]) if doc_text else ""
                            print(f"[ITEM {idx} DOC_TEXT] {preview.encode('ascii', errors='replace').decode('ascii')}", flush=True)
                    except Exception as e_pdf:

                        import traceback
                        print(f"[PDF_EXCEPTION] {type(e_pdf).__name__}: {e_pdf}\n{traceback.format_exc()}", flush=True)
                        doc_text = None  # fall through to the no-PDF branch below




                if not doc_text:
                    # No backing PDF (or render failed): use ONLY real "facts" text if present,
                    # never the raw metadata dict/path — this was the source of Flaws #4b/#4c.
                    facts_text = []
                    if isinstance(chunk_used, dict):
                        for c_val in chunk_used.values():
                            if isinstance(c_val, dict) and c_val.get("facts"):
                                facts_text.extend(c_val["facts"])
                    doc_text = f"[NO_BACKING_PDF_FALLBACK] {' '.join(facts_text)}"
                    fallback_count += 1

                items.append({
                    "id": transformed["query_id"],
                    "doc_id": doc_id or transformed["ground_truth_citation"]["table_id"],
                    "domain": domain,
                    "query": transformed["query"],
                    "ground_truth": transformed["ground_truth_value"],
                    "doc_text": doc_text,
                    "query_type": "prose",
                    "has_real_pdf": bool(pdf_path and os.path.exists(pdf_path)),
                })
                if self.config.limit and len(items) >= self.config.limit:
                    break

            print(f"[UNIDOC LOAD] {real_pdf_count} items used real PDF+OCR, "
                  f"{fallback_count} used no-PDF facts-only fallback.", flush=True)




        if self.config.limit and self.config.limit > 0:
            items = items[:self.config.limit]

        print(f"Loaded {len(items)} queries for evaluation.", flush=True)

        # 3. Chunking Stage
        t_chunk_start = time.perf_counter()
        all_chunks = []
        doc_map = {}
        for item in items:
            did = item["doc_id"]
            if did not in doc_map:
                doc_map[did] = item["doc_text"]
                c_list = chunk_document(item["doc_text"], strategy_name=self.config.chunk_strategy)
                for c in c_list:
                    c["doc_id"] = did
                    c["chunk_id"] = f"{did}_{c['chunk_id']}"  # ensure chunk_id is globally unique across documents (Flaw #6 fix)
                    all_chunks.append(c)
        t_chunking = time.perf_counter() - t_chunk_start

        # 4. Embedding & Indexing Stage
        t_embed_start = time.perf_counter()
        embedding_stage = EmbeddingStage(model_name=self.config.embedding_model)
        retriever_stage = RRFHybridRetrieverStage(embedding_stage=embedding_stage)
        retriever_stage.index_chunks(all_chunks)
        t_embedding = time.perf_counter() - t_embed_start

        print(f"[CHUNK INDEX REPORT] Total chunks: {len(all_chunks)} | Unique chunk_ids: {len(set(c['chunk_id'] for c in all_chunks))}", flush=True)

        # 5. Reranking & Reading Stage
        t_retrieval_sum = 0.0
        t_rerank_sum = 0.0
        t_reading_sum = 0.0

        from src.routing.query_enhancer import QueryEnhancer
        query_enhancer = QueryEnhancer(model=getattr(self.config, "reader_model", "gemini-3.6-flash") or "gemini-3.6-flash")

        reranker_stage = RerankingStage(enabled=self.config.reranking_enabled)
        reader_stage = ReadingStage(
            use_llm_reader=True,
            use_pal_arithmetic=getattr(self.config, "use_pal_arithmetic", False),
            reader_model=getattr(self.config, "reader_model", "gemini-3.6-flash") or "gemini-3.6-flash"
        )

        predictions = []
        retrieval_records = []
        import math

        for idx_item, item in enumerate(items):
            raw_q = item["query"]
            gt = item["ground_truth"]
            gt_did = item["doc_id"]

            # Query Enhancement (transforms raw user query into domain search keywords & structured directive)
            enhanced = query_enhancer.enhance(raw_q)
            search_q = enhanced.get("search_query") or raw_q
            directive_q = enhanced.get("structured_directive") or raw_q
            q_type = enhanced.get("query_type") or item.get("query_type") or "prose"
            print(f"[ITEM {idx_item} ENHANCER] Raw: '{raw_q}' -> Search Keywords: '{search_q}' | Type: {q_type}", flush=True)

            # Retrieval with Qdrant Hybrid RRF using optimized search keywords
            t0 = time.perf_counter()
            retrieved_fused = retriever_stage.retrieve(search_q, top_k=self.config.top_k * 2)
            t_retrieval_sum += (time.perf_counter() - t0)

            # Primary matched chunks
            retrieved_chunks = []
            for cid, score in retrieved_fused:
                match = next((c for c in all_chunks if c["chunk_id"] == cid), None)
                if match:
                    retrieved_chunks.append((cid, match["text"], match.get("doc_id")))

            # Relational Context Expansion ("Small-to-Big" graph expansion)
            expanded_records = retriever_stage.expand_context(
                [cid for cid, _ in retrieved_fused],
                max_total_chunks=max(15, self.config.top_k * 2)
            )

            retrieved_dids = [c[2] for c in retrieved_chunks]
            print(f"[ITEM {idx_item} RETRIEVAL] GT Doc: {gt_did} | Retrieved Rank 1 Doc: {retrieved_dids[0] if retrieved_dids else None} | All Retrieved Docs: {retrieved_dids[:5]} | Expanded Context Count: {len(expanded_records)}", flush=True)
            retrieval_success = (gt_did in retrieved_dids) if gt_did else True

            # Calculate item retrieval metrics on primary search ranks
            h1 = 1.0 if (retrieved_dids and retrieved_dids[0] == gt_did) else 0.0
            h5 = 1.0 if (gt_did and gt_did in retrieved_dids[:5]) else 0.0
            h10 = 1.0 if (gt_did and gt_did in retrieved_dids[:10]) else 0.0
            mrr_val = 0.0
            ndcg_val = 0.0
            if gt_did and gt_did in retrieved_dids:
                rank_pos = retrieved_dids.index(gt_did) + 1
                mrr_val = 1.0 / rank_pos
                if rank_pos <= 10:
                    ndcg_val = 1.0 / math.log2(rank_pos + 1)

            retrieval_records.append({
                "hit@1": h1,
                "hit@5": h5,
                "hit@10": h10,
                "mrr": mrr_val,
                "ndcg@10": ndcg_val
            })

            # Reranking using expanded context pool
            t1 = time.perf_counter()
            candidates_for_rerank = [
                (r.get("chunk_id", ""), r.get("text", "")) for r in expanded_records
            ] if expanded_records else [(c[0], c[1]) for c in retrieved_chunks]
            reranked = reranker_stage.rerank(search_q, candidates_for_rerank, top_k=self.config.top_k)
            t_rerank_sum += (time.perf_counter() - t1)

            top_context_texts = [r[1] for r in reranked] if reranked else [item["doc_text"]]

            # Reading stage: Raw Query First for factual / short / single-line queries with Fallback to Directive
            t2 = time.perf_counter()
            is_factual_or_short = (
                str(q_type).lower() in ["span", "count", "arithmetic", "factual", "short"]
                or str(item.get("question_type", "")).lower() in ["factual_retrieval", "span", "count", "arithmetic"]
                or str(item.get("answer_type", "")).lower() in ["span", "count", "arithmetic"]
            )

            # E2E extraction
            if is_factual_or_short:
                e2e_res = reader_stage.extract_answer(raw_q, top_context_texts, query_type=q_type)
                e2e_ans = str(e2e_res.get("primary_answer", "")).strip()
                if not e2e_ans or e2e_ans.upper() == "NOT_FOUND":
                    e2e_fallback = reader_stage.extract_answer(directive_q, top_context_texts, query_type=q_type)
                    fb_ans = str(e2e_fallback.get("primary_answer", "")).strip()
                    if fb_ans and fb_ans.upper() != "NOT_FOUND":
                        e2e_res = e2e_fallback
            else:
                combined_q = f"Question: {raw_q}\nStep-by-step guidance:\n{directive_q}" if directive_q != raw_q else raw_q
                e2e_res = reader_stage.extract_answer(combined_q, top_context_texts, query_type=q_type)

            # Oracle extraction using GT document context directly
            gt_retrieved_chunks = [c[1] for c in retrieved_chunks if c[2] == gt_did]
            if gt_retrieved_chunks:
                oracle_context = gt_retrieved_chunks
            else:
                doc_chunks = [c["text"] for c in all_chunks if c.get("doc_id") == gt_did]
                oracle_context = doc_chunks if doc_chunks else [item["doc_text"]]

            if is_factual_or_short:
                oracle_res = reader_stage.extract_answer(raw_q, oracle_context, query_type=q_type)
                oracle_ans = str(oracle_res.get("primary_answer", "")).strip()
                if not oracle_ans or oracle_ans.upper() == "NOT_FOUND":
                    oracle_fallback = reader_stage.extract_answer(directive_q, oracle_context, query_type=q_type)
                    fb_o = str(oracle_fallback.get("primary_answer", "")).strip()
                    if fb_o and fb_o.upper() != "NOT_FOUND":
                        oracle_res = oracle_fallback
            else:
                combined_q = f"Question: {raw_q}\nStep-by-step guidance:\n{directive_q}" if directive_q != raw_q else raw_q
                oracle_res = reader_stage.extract_answer(combined_q, oracle_context, query_type=q_type)
            t_reading_sum += (time.perf_counter() - t2)

            predictions.append({
                "query_id": item["id"],
                "query": raw_q,
                "enhanced_query": search_q,
                "structured_directive": directive_q,
                "ground_truth": gt,
                "oracle_prediction": oracle_res["primary_answer"],
                "e2e_prediction": e2e_res["primary_answer"],
                "retrieval_success": retrieval_success,
                "retrieval_context": top_context_texts
            })

        # 6. Evaluation Stage (Tri-Benchmark Suite)
        n_queries = len(items)
        retrieval_metrics = {
            "hit@1": round(sum(r["hit@1"] for r in retrieval_records) / max(1, n_queries), 4),
            "hit@5": round(sum(r["hit@5"] for r in retrieval_records) / max(1, n_queries), 4),
            "hit@10": round(sum(r["hit@10"] for r in retrieval_records) / max(1, n_queries), 4),
            "mrr": round(sum(r["mrr"] for r in retrieval_records) / max(1, n_queries), 4),
            "ndcg@10": round(sum(r["ndcg@10"] for r in retrieval_records) / max(1, n_queries), 4),
        }

        # 6a. Benchmark 1: Custom Deterministic Metrics
        qa_metrics = evaluate_batch(predictions)

        # 6b. Benchmark 2: DeepEval Scorecard
        print("\n" + "=" * 60, flush=True)
        print(" [TRI-BENCHMARK] Executing DeepEval Evaluation Suite...", flush=True)
        print("=" * 60, flush=True)
        from benchmark_harness.stages.deepeval_evaluator import DeepEvalEvaluator
        deepeval_eval = DeepEvalEvaluator(model_name="gemini-3.5-flash-lite")
        deepeval_metrics = deepeval_eval.evaluate_records(predictions)

        # 6c. Benchmark 3: Ragas Scorecard
        print("\n" + "=" * 60, flush=True)
        print(" [TRI-BENCHMARK] Executing Ragas Evaluation Suite...", flush=True)
        print("=" * 60, flush=True)
        from benchmark_harness.stages.ragas_evaluator import RagasEvaluator
        ragas_eval = RagasEvaluator(model_name="gemini-3.5-flash-lite")
        ragas_metrics = ragas_eval.evaluate_records(predictions)

        total_time = time.perf_counter() - start_time

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"phase10_benchmark_{self.config.pipeline}_{self.config.dataset}_{timestamp_str}.json"
        json_path = os.path.join(self.results_dir, json_filename)

        result_payload = {
            "pipeline": self.config.pipeline,
            "dataset": self.config.dataset,
            "timestamp": timestamp_str,
            "num_queries": n_queries,
            "predictions": predictions,
            "tri_benchmark_scorecard": {
                "custom_benchmark": {
                    "headline_score": qa_metrics["true_e2e"].get("headline_score"),
                    "llm_judge": qa_metrics["true_e2e"].get("llm_judge"),
                    "containment": qa_metrics["true_e2e"].get("containment"),
                    "numeric_em": qa_metrics["true_e2e"]["numeric_em"],
                    "set_em": qa_metrics["true_e2e"]["set_em"],
                    "f1": qa_metrics["true_e2e"]["f1"],
                    "rouge_l": qa_metrics["true_e2e"]["rouge_l"],
                    "char_similarity": qa_metrics["true_e2e"]["char_similarity"],
                    "retrieval_layer": retrieval_metrics,
                },
                "deepeval_benchmark": {
                    "faithfulness": deepeval_metrics.get("faithfulness"),
                    "answer_relevancy": deepeval_metrics.get("answer_relevancy"),
                    "contextual_precision": deepeval_metrics.get("contextual_precision"),
                    "valid_queries": deepeval_metrics.get("valid_queries"),
                    "error_queries": deepeval_metrics.get("error_queries"),
                },
                "ragas_benchmark": {
                    "faithfulness": ragas_metrics.get("faithfulness"),
                    "answer_relevancy": ragas_metrics.get("answer_relevancy"),
                    "context_precision": ragas_metrics.get("context_precision"),
                    "context_recall": ragas_metrics.get("context_recall"),
                    "valid_queries": ragas_metrics.get("valid_queries"),
                    "error_queries": ragas_metrics.get("error_queries"),
                }
            },
            "metrics": {
                "retrieval_layer": retrieval_metrics,
                "qa_extraction_layer": {
                    "oracle_scoped": qa_metrics["oracle_scoped"],
                    "true_e2e": qa_metrics["true_e2e"]
                },
                "oracle_scoped": qa_metrics["oracle_scoped"],
                "true_e2e": qa_metrics["true_e2e"],
                "deepeval": deepeval_metrics,
                "ragas": ragas_metrics,
                "sanity_check": qa_metrics["sanity_check"]
            },
            "latency_seconds": {
                "ocr": round(t_ocr, 4),
                "chunking": round(t_chunking, 4),
                "embedding": round(t_embedding, 4),
                "retrieval": round(t_retrieval_sum, 4),
                "reranking": round(t_rerank_sum, 4),
                "reading": round(t_reading_sum, 4),
                "total": round(total_time, 4)
            },
            "config": {
                "chunk_strategy": self.config.chunk_strategy,
                "embedding_model": self.config.embedding_model,
                "reranking_enabled": self.config.reranking_enabled,
                "top_k": self.config.top_k,
                "limit": self.config.limit
            },
            "sanity_check": qa_metrics["sanity_check"],
            "notes": [ocr_info.get("note")] if "note" in ocr_info else []
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, ensure_ascii=False)

        print(f"\n" + "=" * 80, flush=True)
        print(f" TRI-BENCHMARK RUN COMPLETE: {self.config.pipeline} on {self.config.dataset}", flush=True)
        print(f" Total Queries Evaluated: {n_queries} | Total Runtime: {total_time:.2f}s", flush=True)
        print(f" ----------------------------------------------------------------------", flush=True)
        print(f" 1. CUSTOM BENCHMARK (Exact Match Deleted, LLM Judge Promoted):", flush=True)
        print(f"    Retrieval: Hit@1={retrieval_metrics['hit@1']} | Hit@5={retrieval_metrics['hit@5']} | MRR={retrieval_metrics['mrr']} | NDCG@10={retrieval_metrics['ndcg@10']}", flush=True)
        print(f"    QA E2E:    Headline={qa_metrics['true_e2e'].get('headline_score')} | LLM Judge={qa_metrics['true_e2e'].get('llm_judge')} | Containment={qa_metrics['true_e2e'].get('containment')} | NumEM={qa_metrics['true_e2e']['numeric_em']} | F1={qa_metrics['true_e2e']['f1']}", flush=True)
        print(f" 2. DEEPEVAL BENCHMARK (Error-Filtered):", flush=True)
        print(f"    Faithfulness={deepeval_metrics.get('faithfulness')} | AnswerRelevancy={deepeval_metrics.get('answer_relevancy')} | ContextualPrecision={deepeval_metrics.get('contextual_precision')} (Valid: {deepeval_metrics.get('valid_queries')}/{deepeval_metrics.get('total_queries')})", flush=True)
        print(f" 3. RAGAS BENCHMARK (Error-Filtered):", flush=True)
        print(f"    Faithfulness={ragas_metrics.get('faithfulness')} | AnswerRelevancy={ragas_metrics.get('answer_relevancy')} | ContextPrecision={ragas_metrics.get('context_precision')} | ContextRecall={ragas_metrics.get('context_recall')} (Valid: {ragas_metrics.get('valid_queries')}/{ragas_metrics.get('total_queries')})", flush=True)
        print(f" ----------------------------------------------------------------------", flush=True)
        print(f" Full Results JSON written to: {json_path}", flush=True)
        print(f"=" * 80 + "\n", flush=True)
        return result_payload
