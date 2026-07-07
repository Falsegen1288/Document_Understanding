"""
main.py
-------
Master Document Understanding Pipeline Orchestrator.
Runs end-to-end PDF layout segmentation, text extraction, OCR, table parsing, and visual figure captions.
"""

import os
import sys
import argparse
import json
import time
import yaml
import fitz  # PyMuPDF
from PIL import Image

# Add root folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from algorithms.config import OUTPUT_DIR

# ── BBox Drawing Colors ──────────────────────────────────────────────────────
COLOR_MAP = {
    "text": (0.13, 0.59, 0.95),            # Blue
    "title": (1.0, 0.34, 0.13),            # Orange
    "table": (0.61, 0.15, 0.69),            # Purple
    "figure": (0.3, 0.69, 0.31),            # Green
    "image": (0.3, 0.69, 0.31),             # Green
    "caption": (0.0, 0.74, 0.83),           # Teal/Cyan
    "list": (1.0, 0.6, 0.0),                # Amber
    "section_header": (0.91, 0.12, 0.39),   # Pink
    "footnote": (0.47, 0.33, 0.28),         # Brown
    "formula": (0.57, 0.57, 0.57),          # Dark Grey
    "footer": (0.38, 0.49, 0.55),           # Blue Grey
    "header": (0.38, 0.49, 0.55)            # Blue Grey
}
DEFAULT_COLOR = (0.62, 0.62, 0.62)

# ── Load Config Helper ────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load default algorithm configurations from config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.isfile(config_path):
        return {
            "layout_detection": "doclayout_yolo",
            "text_extraction": "pymupdf",
            "ocr": "paddleocr",
            "table_extraction": "docling_tableformer",
            "image_extraction": "gemini"
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ── Context-Grounded Captioning Helpers ────────────────────────────────────────

def _pre_extract_page_text(page_no, detected_elements, doc, pdf_path, cfg, has_digital_text, page_images):
    """Pre-extract/OCR text for all non-visual elements on a specific page."""
    print(f"  [CONTEXT] Pre-extracting text/OCR context for page {page_no}...")
    for element in detected_elements:
        if element["page"] != page_no:
            continue
        el_type = element["type"]
        if el_type in ["table", "figure", "image"]:
            continue
        
        content = element.get("content", "").strip()
        if not content:
            bbox = element["bbox"]
            if has_digital_text:
                fitz_page = doc[page_no - 1]
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                if cfg["text_extraction"] == "pymupdf":
                    content = fitz_page.get_text("text", clip=rect).strip()
                else:
                    import pdfplumber
                    try:
                        with pdfplumber.open(pdf_path) as pl:
                            pl_page = pl.pages[page_no - 1]
                            cropped = pl_page.within_bbox((bbox[0], bbox[1], bbox[2], bbox[3]))
                            content = cropped.extract_text() or ""
                    except Exception:
                        content = ""
            else:
                pdf_w = doc[page_no - 1].rect.width
                pdf_h = doc[page_no - 1].rect.height
                page_img = Image.open(page_images[page_no - 1])
                img_w, img_h = page_img.size
                
                crop_box = [
                    bbox[0] * (img_w / pdf_w),
                    bbox[1] * (img_h / pdf_h),
                    bbox[2] * (img_w / pdf_w),
                    bbox[3] * (img_h / pdf_h)
                ]
                cropped_img = page_img.crop(crop_box)
                
                try:
                    if cfg["ocr"] == "tesseract":
                        from algorithms.text_extraction.scanned.tesseract.extractor import extract_text as tesseract_ocr
                        res = tesseract_ocr(cropped_img)
                    elif cfg["ocr"] == "easyocr":
                        from algorithms.text_extraction.scanned.easyocr.extractor import extract_text as easy_ocr
                        res = easy_ocr(cropped_img)
                    else:
                        from algorithms.text_extraction.scanned.paddleocr.extractor import extract_text as paddle_ocr
                        res = paddle_ocr(cropped_img)
                    content = res.get("full_text", "").strip()
                except Exception as ocr_err:
                    print(f"    [WARNING] Pre-extraction OCR failed: {ocr_err}")
                    content = ""
            element["content"] = content

def merge_tables_into_elements(page_no, page_tables, detected_elements):
    """Associate extracted table datasets with their corresponding bounding box elements."""
    table_elements = [el for el in detected_elements if el["page"] == page_no and el["type"] == "table"]
    
    if not table_elements or not page_tables:
        return
        
    import pandas as pd
    # Match each extracted table to the closest table bounding box on the page
    for t_idx, table in enumerate(page_tables):
        t_bbox = table.get("bbox", [0, 0, 0, 0])
        t_cx = (t_bbox[0] + t_bbox[2]) / 2
        t_cy = (t_bbox[1] + t_bbox[3]) / 2
        
        best_dist = float("inf")
        best_el = None
        
        for el in table_elements:
            if "extracted" in el:
                continue
            el_bbox = el["bbox"]
            el_cx = (el_bbox[0] + el_bbox[2]) / 2
            el_cy = (el_bbox[1] + el_bbox[3]) / 2
            
            dist = (t_cx - el_cx)**2 + (t_cy - el_cy)**2
            if dist < best_dist:
                best_dist = dist
                best_el = el
                
        if not best_el and table_elements:
            best_el = table_elements[0]
            
        if best_el:
            table["matched"] = True
            best_el["extracted"] = {
                "type": "table",
                "markdown": table.get("markdown", ""),
                "dataframe_csv": pd.DataFrame(table.get("data", [])).to_csv(index=False) if table.get("data") else "",
                "extractor": table.get("engine", "tatr") or "tatr",
                "rows": len(table.get("data", [])),
                "cols": len(table.get("data", [[]])[0]) if table.get("data") else 0
            }

def compile_grounded_context_and_prompt(page_no, figure_bbox, detected_elements, pdf_path):
    """Compile hierarchical layout-aware context and build domain-specific VLM prompt."""
    # 1. Document Title
    document_title = "Document"
    for el in detected_elements:
        if el["type"] == "title" and el.get("content", "").strip():
            document_title = el["content"].strip()
            break
            
    import numpy as np
    
    # ── Geometric Primitive Helpers ──────────────────────────────────────────
    def cx(box):  return (box[0] + box[2]) / 2
    def cy(box):  return (box[1] + box[3]) / 2
    def w(box):   return max(1.0, box[2] - box[0])
    def h(box):   return max(1.0, box[3] - box[1])
    
    def overlap_1d(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))
        
    def x_overlap_ratio(boxA, boxB):
        ov = overlap_1d(boxA[0], boxA[2], boxB[0], boxB[2])
        return ov / min(w(boxA), w(boxB))
        
    def y_overlap_ratio(boxA, boxB):
        ov = overlap_1d(boxA[1], boxA[3], boxB[1], boxB[3])
        return ov / min(h(boxA), h(boxB))
        
    def horizontal_gap(boxA, boxB):
        if boxB[0] >= boxA[2]:  # B is to the right of A
            return boxB[0] - boxA[2]
        elif boxA[0] >= boxB[2]:  # B is to the left of A
            return boxA[0] - boxB[2]
        else:
            return 0.0  # Overlapping horizontally
            
    # Page dimension estimates from elements to guide limits
    page_w = 612.0
    page_h = 792.0
    for el in detected_elements:
        if el["page"] == page_no:
            page_w = max(page_w, el["bbox"][2])
            page_h = max(page_h, el["bbox"][3])
            
    fig_cx = cx(figure_bbox)
    fig_cy = cy(figure_bbox)
    fig_w = w(figure_bbox)
    fig_h = h(figure_bbox)
    fig_y0 = figure_bbox[1]
    
    # ── 1. Heading/Caption Association ───────────────────────────────────────
    HEADING_TYPES = {'figure_caption', 'table_caption', 'section_header', 'title', 'header'}
    best_heading_score = -1.0
    best_heading_el = None
    
    for el in detected_elements:
        if el["page"] != page_no or el["type"] not in HEADING_TYPES:
            continue
            
        el_bbox = el["bbox"]
        # Hard Filter A: C must be above F (with a slight 10pt buffer for overlay offsets)
        if el_bbox[3] > fig_y0 + 10:
            continue
        # Hard Filter B: C must not be too far above (max 30% of page height)
        dy = fig_y0 - el_bbox[3]
        if dy > 0.30 * page_h:
            continue
        # Hard Filter C: Require minimum column overlap (at least 15% of width)
        col_ov = x_overlap_ratio(figure_bbox, el_bbox)
        if col_ov < 0.15:
            continue
            
        drift_x = abs(fig_cx - cx(el_bbox))
        
        # Soft Score: Combine column overlap, vertical gap, and horizontal column drift
        # beta/alpha = 3 -> Horizontal column drift is penalized 3x more heavily than vertical gap
        alpha = 1.0
        beta = 3.0
        
        raw_score = col_ov / (1.0 + alpha * (dy / fig_h) + beta * (drift_x / fig_w))
        
        # Type priority multiplier
        type_preference = {
            'figure_caption': 2.0,
            'table_caption': 1.5,
            'section_header': 1.0,
            'title': 0.5,
            'header': 0.3
        }
        tau = type_preference.get(el["type"], 1.0)
        score = tau * raw_score
        
        if score > best_heading_score:
            best_heading_score = score
            best_heading_el = el

    # Relaxed Fallback: catch wide headings/titles
    if best_heading_el is None:
        best_heading_score = -1.0
        for el in detected_elements:
            if el["page"] != page_no or el["type"] not in HEADING_TYPES:
                continue
            el_bbox = el["bbox"]
            if el_bbox[3] > fig_y0 + 15:
                continue
            col_ov = x_overlap_ratio(figure_bbox, el_bbox)
            # Just look at center-drift and vertical gap without column constraint
            dy = fig_y0 - el_bbox[3]
            drift_x = abs(fig_cx - cx(el_bbox))
            score = 1.0 / (1.0 + (dy / fig_h) + (drift_x / fig_w))
            
            tau = {'figure_caption': 2.0, 'table_caption': 1.5, 'section_header': 1.0, 'title': 0.8}.get(el["type"], 0.5)
            score = tau * score
            if score > best_heading_score:
                best_heading_score = score
                best_heading_el = el

    nearest_heading = "N/A"
    if best_heading_el:
        nearest_heading = best_heading_el.get("content", "").strip()

    # Fallback to main page title if still N/A
    if nearest_heading == "N/A" or not nearest_heading:
        for el in detected_elements:
            if el["page"] == page_no and el["type"] == "title" and el.get("content", "").strip():
                nearest_heading = el["content"].strip()
                break

    # ── 2. Adjacent Table Association ────────────────────────────────────────
    best_table_score = -1.0
    best_table_el = None
    
    for el in detected_elements:
        if el["page"] != page_no or el["type"] != "table":
            continue
            
        el_bbox = el["bbox"]
        # Hard Filter A: Must vertically co-occur on the same "row" (overlap > 25%)
        y_ov = y_overlap_ratio(figure_bbox, el_bbox)
        if y_ov < 0.25:
            continue
        # Hard Filter B: Must not horizontally overlay (must be left or right)
        if x_overlap_ratio(figure_bbox, el_bbox) >= 0.15:
            continue
        # Hard Filter C: Horizontal gap must be within threshold (max 20% of page width)
        dx = horizontal_gap(figure_bbox, el_bbox)
        if dx > 0.20 * page_w:
            continue
            
        dy_center = abs(fig_cy - cy(el_bbox))
        sigma_px = 0.06 * page_w
        
        # Soft Score: exponential decay for horizontal gap, penalized center misalignment
        score = y_ov * np.exp(-dx / sigma_px) * (1.0 / (1.0 + dy_center / fig_h))
        
        if score > best_table_score:
            best_table_score = score
            best_table_el = el

    table_markdown = ""
    if best_table_el:
        ext = best_table_el.get("extracted", {})
        if ext and ext.get("type") == "table":
            table_markdown = ext.get("markdown", "")
        if not table_markdown:
            table_markdown = f"Table Region at [{int(best_table_el['bbox'][0])}, {int(best_table_el['bbox'][1])}]"

    # ── 3. Adjacent Description Paragraph ────────────────────────────────────
    best_para_score = -1.0
    best_para_el = None
    
    for el in detected_elements:
        if el["page"] != page_no or el["type"] not in {'plain text', 'list', 'text'}:
            continue
        if best_heading_el and el.get("id") == best_heading_el.get("id"):
            continue
            
        el_bbox = el["bbox"]
        col_ov = x_overlap_ratio(figure_bbox, el_bbox)
        if col_ov < 0.15:
            continue
            
        if el_bbox[1] >= figure_bbox[3]:  # Below F (preferred)
            dy = el_bbox[1] - figure_bbox[3]
            d_bias = 1.0
        elif el_bbox[3] <= figure_bbox[1]:  # Above F
            dy = figure_bbox[1] - el_bbox[3]
            d_bias = 0.5
        else:
            continue
            
        score = d_bias * col_ov / (1.0 + dy / fig_h)
        if score > best_para_score:
            best_para_score = score
            best_para_el = el

    adjacent_paragraph = ""
    if best_para_el:
        adjacent_paragraph = best_para_el.get("content", "").strip()

    # ── 4. Build Context Representations ─────────────────────────────────────
    page_text_blocks = []
    for el in detected_elements:
        if el["page"] == page_no and el["type"] not in ["table", "figure", "image"]:
            content = el.get("content", "").strip()
            if content:
                page_text_blocks.append(content)
    ocr_text_same_page = " ".join(page_text_blocks)

    # Key Term Seeding
    import re
    words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]{2,}\b|\b\d+[\.\d]*\s*[a-zA-Z]+\b', ocr_text_same_page)
    seen = set()
    key_terms = []
    for w in words:
        wl = w.lower()
        if wl not in seen and len(wl) > 3:
            seen.add(wl)
            key_terms.append(w)
            if len(key_terms) >= 15:
                break
    terms_str = ", ".join(key_terms) if key_terms else "None detected"

    # Domain Classification
    path_lower = pdf_path.lower()
    text_lower = ocr_text_same_page.lower()
    
    is_medical = "medical" in path_lower or any(w in text_lower for w in ["patient", "clinical", "forceps", "surgical", "treatment", "doctor", "health", "hospital", "anatomy"])
    is_scientific = "scientific" in path_lower or "arxiv" in path_lower or any(w in text_lower for w in ["abstract", "equation", "theorem", "method", "dataset", "framework", "architecture", "experimental"])
    is_financial = "financial" in path_lower or any(w in text_lower for w in ["fiscal", "quarter", "revenue", "profit", "ebitda", "shares", "growth", "annual report", "balance sheet"])
    
    if is_medical:
        domain = "Medical / Clinical"
        domain_inst = "Describe what this figure shows in highly precise clinical and technical terms. Reference any medical instruments, structures, patient metrics, or anatomical terms consistent with the surrounding text context."
    elif is_scientific:
        domain = "Scientific / Academic"
        domain_inst = "Describe this figure using the academic terminology and notation of the paper. Focus on clarifying what the diagram, chart, or framework illustrates in a way that complements the paper's methodology."
    elif is_financial:
        domain = "Financial / Business"
        domain_inst = "Describe what this business figure or chart shows in precise corporate finance terms. Capture metric names, fiscal trends, or growth comparisons consistent with the financial report context."
    else:
        domain = "General / Technical"
        domain_inst = "Describe this figure in a way that is consistent with the surrounding terminology and context of the page."

    # Compile rich layout-aware grounded context text
    grounded_context = f"### TARGET PRODUCT HEADING:\n{nearest_heading}\n\n"
    if table_markdown:
        grounded_context += f"### ADJACENT SPECIFICATIONS TABLE:\n{table_markdown}\n\n"
    if adjacent_paragraph:
        grounded_context += f"### LOCAL PRODUCT DESCRIPTION:\n{adjacent_paragraph}\n\n"
    
    if not table_markdown and not adjacent_paragraph:
        grounded_context += f"### SURROUNDING PAGE TEXT:\n{ocr_text_same_page[:1000]}"

    # Assemble structured VLM prompt
    prompt = f"""You are a document understanding assistant. Describe the provided figure crop so that the caption is semantically anchored to the document's own vocabulary, headings, and data tables.

[SURROUNDING GEOMETRIC CONTEXT]
- Document Title: {document_title}
- Document Domain: {domain}
- Target Local Heading: {nearest_heading}
"""
    if table_markdown:
        prompt += f"- Adjacent Specifications Table:\n{table_markdown}\n"
    if adjacent_paragraph:
        prompt += f"- Local Product Description:\n{adjacent_paragraph}\n"
    prompt += f"- Key Terms from Page: {terms_str}\n"
    prompt += f"""
[INSTRUCTION]
{domain_inst}
Do not repeat what the text already says; instead, focus on describing what the figure adds visually (elements, loops, labels, axes, trends, or structures).
Refer to exact product codes or sizing from the table if visible in the image crop, but maintain focus on visual verification.
Keep the final caption concise and professional (2-3 sentences max).
"""
    return prompt, grounded_context, nearest_heading, domain, best_table_el

def compute_semantic_similarity(caption, context_text):
    """Compute semantic correlation score between caption and page context."""
    if not caption or not context_text:
        return 0.0
        
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        encoded = tokenizer([caption, context_text[:1000]], padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded)
        embeddings = mean_pooling(model_output, encoded['attention_mask'])
        
        emb1 = embeddings[0] / torch.norm(embeddings[0])
        emb2 = embeddings[1] / torch.norm(embeddings[1])
        
        score = float(torch.dot(emb1, emb2).item())
        print(f"  [SIMILARITY] Computed neural cosine similarity: {score:.3f}")
        return score
    except Exception:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
            tfidf = vectorizer.fit_transform([caption, context_text])
            pairwise_similarity = (tfidf * tfidf.T).toarray()
            score = float(pairwise_similarity[0, 1])
            print(f"  [SIMILARITY] Computed TF-IDF character similarity (fallback): {score:.3f}")
            return score
        except Exception:
            words1 = set(caption.lower().split())
            words2 = set(context_text.lower().split())
            if not words1 or not words2:
                return 0.0
            score = len(words1.intersection(words2)) / min(len(words1), len(words2))
            print(f"  [SIMILARITY] Computed basic Jaccard intersection score: {score:.3f}")
            return score

def critique_and_rewrite_caption(image, initial_caption, context_text, engine, api_key=None):
    """Two-pass VLM critique loop to rewrite caption for better alignment."""
    print("  [CRITIQUE] Similarity score below threshold. Triggering Second-Pass VLM critique...")
    
    critique_prompt = f"""You are a document understanding assistant. The following caption generated for this figure crop does not sufficiently align with the terminology and vocabulary of the surrounding document text.

[SURROUNDING TEXT CONTEXT]
{context_text[:1000]}

[INITIAL CAPTION]
{initial_caption}

[INSTRUCTION]
Critique and rewrite the caption. Ensure the rewritten caption is:
1. Highly descriptive of the visual components (axes, labels, diagrams, or images).
2. Semantically anchored to the surrounding text context and exact vocabulary of the page.
3. Concise and professional (2-3 sentences).

Return ONLY the rewritten final caption. Do not include any intro, outro, or explanations.
"""
    try:
        if engine == "groq_llama":
            from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
            res = groq_desc(image, prompt=critique_prompt, api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
            return res.get("description", initial_caption)
        elif engine == "groq_qwen":
            from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
            res = groq_desc(image, prompt=critique_prompt, api_key=api_key, model="qwen/qwen3.6-27b")
            return res.get("description", initial_caption)
        elif engine == "local_qwen":
            from algorithms.image_extraction.local.extractor import describe_figure as local_desc
            res = local_desc(image, prompt=critique_prompt, model="qwen2.5vl:3b")
            return res.get("description", initial_caption)
        elif engine == "local_moondream":
            from algorithms.image_extraction.local.extractor import describe_figure as local_desc
            res = local_desc(image, prompt=critique_prompt, model="moondream:latest")
            return res.get("description", initial_caption)
        else:
            from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
            res = groq_desc(image, prompt=critique_prompt, api_key=api_key)
            return res.get("description", initial_caption)
    except Exception as err:
        print(f"  [WARNING] Critique loop failed: {err}. Retaining initial caption.")
    return initial_caption


# ── Master Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(pdf_path: str, output_root: str, overrides: dict) -> str:
    """Run document extraction on a PDF using configured algorithms."""
    start_time = time.time()
    doc_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc_output_dir = os.path.join(output_root, doc_stem)
    os.makedirs(doc_output_dir, exist_ok=True)
    
    # A1-Lite Ingestion Manifest generation and duplicate check
    from ingestion import generate_manifest, find_existing_manifest
    manifest = generate_manifest(pdf_path)
    content_hash = manifest["content_hash"]
    doc_id = manifest["doc_id"]
    
    # Check if identical file (same content_hash) has already been processed
    existing_doc_id, existing_manifest_path = find_existing_manifest(content_hash, output_root)
    if existing_doc_id and os.path.exists(os.path.dirname(existing_manifest_path)):
        print(f"\n[INGESTION] Document already ingested! content_hash: {content_hash} | doc_id: {existing_doc_id}")
        print(f"[INGESTION] Skipping pipeline run for {pdf_path}")
        return os.path.dirname(existing_manifest_path)
        
    # Check if same filename but different content_hash exists (edited/updated file)
    manifest_filename = os.path.join(doc_output_dir, "manifest.json")
    if os.path.exists(manifest_filename):
        try:
            with open(manifest_filename, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
            if old_manifest.get("content_hash") != content_hash:
                print(f"\n[INGESTION] [IMPORTANT] Updated version of previously seen filename detected: {os.path.basename(pdf_path)}")
                print(f"  Old doc_id: {old_manifest.get('doc_id')} | New doc_id: {doc_id}")
        except Exception:
            pass
            
    # Save the manifest to outputs/{doc_stem}/manifest.json
    with open(manifest_filename, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [INGESTION] Generated manifest for {os.path.basename(pdf_path)} -> doc_id: {doc_id}")

    
    print("\n" + "=" * 60)
    print(f" PIPELINE EXECUTION FOR: {doc_stem}")
    print(f" Source PDF: {pdf_path}")
    print(f" Output Dir: {doc_output_dir}")
    print("=" * 60)

    # 1. Load & merge configuration options
    cfg = load_config()
    for k, v in overrides.items():
        if v:
            cfg[k] = v

    print("\n[CONFIG] Executing steps with:")
    for k, v in cfg.items():
        print(f"  - {k:<18}: {v}")

    # Render PDF pages to working image frames
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    
    print("\n[STEP 1] Rendering PDF pages to frames...")
    page_images = []
    for idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_path = os.path.join(doc_output_dir, f"page_{idx+1}.png")
        pix.save(img_path)
        page_images.append(img_path)
    print(f"  [OK] Rendered {len(page_images)} pages.")

    # Check for digitally embedded text
    has_digital_text = False
    for page in doc:
        if page.get_text("text").strip():
            has_digital_text = True
            break
    print(f"  [INFO] Native embedded text found: {has_digital_text}")

    # 2. Layout Detection
    print("\n[STEP 2] Running layout detection...")
    detected_elements = []
    
    layout_engine = cfg["layout_detection"]
    if layout_engine == "doclayout_yolo":
        from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as layout_detect
    elif layout_engine == "nemotron_parse":
        from algorithms.layout_detection.nemotron_parse.extractor import detect_layout as layout_detect
    elif layout_engine == "landingai_ade":
        from algorithms.layout_detection.landingai_ade.extractor import detect_layout as layout_detect
    else:
        from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as layout_detect
        
    for p_idx, img_path in enumerate(page_images):
        img = Image.open(img_path)
        res = layout_detect(img)
        
        # Coordinate scaling: Map image pixels back to PDF points
        pdf_w = doc[p_idx].rect.width
        pdf_h = doc[p_idx].rect.height
        img_w, img_h = img.size
        scale_x = img_w / pdf_w
        scale_y = img_h / pdf_h
        
        for r in res:
            r["page"] = p_idx + 1
            r["content"] = ""
            # Map coordinates
            r["bbox"] = [
                r["bbox"][0] / scale_x,
                r["bbox"][1] / scale_y,
                r["bbox"][2] / scale_x,
                r["bbox"][3] / scale_y
            ]
            detected_elements.append(r)
            
    # Step 1: Deduplicate overlapping elements
    print(f"\n[DEDUPLICATION] Running element deduplication...")
    before_count = len(detected_elements)
    
    def compute_iou(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - intersection_area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    by_page = {}
    for el in detected_elements:
        by_page.setdefault(el["page"], []).append(el)
        
    deduped_elements = []
    for page_no, page_els in by_page.items():
        dropped_indices = set()
        for i in range(len(page_els)):
            if i in dropped_indices:
                continue
            for j in range(i + 1, len(page_els)):
                if j in dropped_indices:
                    continue
                box1 = page_els[i]["bbox"]
                box2 = page_els[j]["bbox"]
                identical = (box1 == box2)
                iou = compute_iou(box1, box2)
                if identical or iou > 0.85:
                    conf1 = page_els[i].get("confidence", 0.0)
                    conf2 = page_els[j].get("confidence", 0.0)
                    if conf1 >= conf2:
                        dropped_indices.add(j)
                    else:
                        dropped_indices.add(i)
                        break
        for idx, el in enumerate(page_els):
            if idx not in dropped_indices:
                deduped_elements.append(el)
    detected_elements = deduped_elements
    after_count = len(detected_elements)
    print(f"  [DEDUPLICATION] Before: {before_count} | After: {after_count} | Removed: {before_count - after_count} elements.")
    print(f"  [OK] Bounding box segmentation detected {len(detected_elements)} elements using {layout_engine}.")



    # 3. Table Extraction
    print("\n[STEP 3] Running table grid extraction...")
    extracted_tables = []
    
    table_engine = cfg["table_extraction"]
    if table_engine == "docling_tableformer":
        from algorithms.table_extraction.docling_tableformer.extractor import extract_tables as tf_extract
        # Identify page numbers containing detected table regions
        table_pages = sorted(list(set(el["page"] for el in detected_elements if el["type"] == "table")))
        if table_pages:
            print(f"  [INFO] Layout analysis identified table bounding boxes on page(s): {table_pages}. Restricting Docling to these pages.")
            extracted_tables = tf_extract(pdf_path, pages=table_pages)
        else:
            print("  [INFO] No table layout regions detected. Skipping table grid extraction to save time.")
            extracted_tables = []
    else:
        # TATR
        from algorithms.table_extraction.tatr.extractor import extract_tables as tatr_extract
        for p_idx, img_path in enumerate(page_images):
            img = Image.open(img_path)
            res = tatr_extract(img)
            for t in res:
                t["page"] = p_idx + 1
                t["table_index"] = len(extracted_tables) + 1
                extracted_tables.append(t)



            
    print(f"  [OK] Reconstructed {len(extracted_tables)} tables.")

    # Merge extracted tables into detected elements so they are available for visual grounded context
    for p_no in range(1, page_count + 1):
        page_tables = [t for t in extracted_tables if t.get("page") == p_no]
        merge_tables_into_elements(p_no, page_tables, detected_elements)

    # 4. Image Visual Captions
    print("\n[STEP 4] Running image vision captions with Context-Grounded Prompting...")
    extracted_figures = []
    
    # Locate layout picture bounding boxes and run vision descriptions
    for element in detected_elements:
        if element["type"] in ["figure", "image"]:
            page_no = element["page"]
            bbox = element["bbox"]
            
            # Crop image region
            pdf_w = doc[page_no - 1].rect.width
            pdf_h = doc[page_no - 1].rect.height
            page_img = Image.open(page_images[page_no - 1])
            img_w, img_h = page_img.size
            
            crop_box = [
                bbox[0] * (img_w / pdf_w),
                bbox[1] * (img_h / pdf_h),
                bbox[2] * (img_w / pdf_w),
                bbox[3] * (img_h / pdf_h)
            ]
            cropped_img = page_img.crop(crop_box)
            fig_crop_filename = f"page_{page_no}_figure_{int(bbox[0])}_{int(bbox[1])}.png"
            fig_crop_path = os.path.join(doc_output_dir, fig_crop_filename)
            try:
                cropped_img.save(fig_crop_path)
            except Exception as fig_err:
                print(f"    [WARNING] Figure crop saving failed: {fig_err}")

            
            # Context pre-extraction for the current page
            _pre_extract_page_text(page_no, detected_elements, doc, pdf_path, cfg, has_digital_text, page_images)
            
            # Compile grounded prompt
            grounded_prompt, context_text, heading, domain, matched_table = compile_grounded_context_and_prompt(
                page_no, bbox, detected_elements, pdf_path
            )
            print(f"  [CAPTIONING] Page {page_no} figure | Domain: {domain} | Heading: '{heading}'")
            
            table_crop_path = ""
            if matched_table:
                try:
                    t_bbox = matched_table["bbox"]
                    t_crop_box = [
                        t_bbox[0] * (img_w / pdf_w),
                        t_bbox[1] * (img_h / pdf_h),
                        t_bbox[2] * (img_w / pdf_w),
                        t_bbox[3] * (img_h / pdf_h)
                    ]
                    cropped_t = page_img.crop(t_crop_box)
                    t_crop_filename = f"page_{page_no}_table_{int(bbox[0])}_{int(bbox[1])}.png"
                    t_crop_path = os.path.join(doc_output_dir, t_crop_filename)
                    cropped_t.save(t_crop_path)
                    table_crop_path = f"outputs/{doc_stem}/{t_crop_filename}"
                except Exception as t_err:
                    print(f"    [WARNING] Table cropping failed: {t_err}")
            
            # Call vision caption models with the grounded prompt
            desc_text = ""
            engine = cfg["image_extraction"]
            
            if engine == "groq_llama":
                from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
                res = groq_desc(cropped_img, prompt=grounded_prompt, model="meta-llama/llama-4-scout-17b-16e-instruct")
                desc_text = res.get("description", "")
            elif engine == "groq_qwen":
                from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
                res = groq_desc(cropped_img, prompt=grounded_prompt, model="qwen/qwen3.6-27b")
                desc_text = res.get("description", "")
            elif engine == "local_qwen":
                from algorithms.image_extraction.local.extractor import describe_figure as local_desc
                res = local_desc(cropped_img, prompt=grounded_prompt, model="qwen2.5vl:3b")
                desc_text = res.get("description", "")
            elif engine == "local_moondream":
                from algorithms.image_extraction.local.extractor import describe_figure as local_desc
                res = local_desc(cropped_img, prompt=grounded_prompt, model="moondream:latest")
                desc_text = res.get("description", "")
            else:
                from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
                res = groq_desc(cropped_img, prompt=grounded_prompt)
                desc_text = res.get("description", "")

                
            # Compute semantic similarity score between caption and page text
            sim_score = compute_semantic_similarity(desc_text, context_text)
            
            # Two-pass verification and rewrite loop if similarity is low
            if sim_score < 0.20 and context_text.strip():
                desc_text = critique_and_rewrite_caption(cropped_img, desc_text, context_text, engine)
                # Re-compute similarity after critique rewrite
                sim_score = compute_semantic_similarity(desc_text, context_text)
                
            extracted_figures.append({
                "page": page_no,
                "bbox": bbox,
                "caption": desc_text,
                "similarity_score": round(sim_score, 3),
                "domain": domain,
                "heading": heading,
                "table_crop_path": table_crop_path,
                "image_path": fig_crop_path
            })

            # Cache visual caption as element content
            element["content"] = desc_text
            
    print(f"  [OK] Generated grounded vision captions for {len(extracted_figures)} figures.")

    # 5. Routing Text Blocks & OCR
    print("\n[STEP 5] Routing text blocks (Native digital vs OCR)...")
    final_elements = []
    
    # Lazy load specific text extraction/OCR modules when actually needed in branches below

    for element in detected_elements:
        el_type = element["type"]
        page_no = element["page"]
        bbox = element["bbox"]
        
        if el_type in ["table", "figure", "image"]:
            final_elements.append(element)
            continue
            
        content = element.get("content", "").strip()
        
        if not content:
            if has_digital_text:
                # Extract native text from region bounding box coordinates
                fitz_page = doc[page_no - 1]
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                if cfg["text_extraction"] == "pymupdf":
                    content = fitz_page.get_text("text", clip=rect).strip()
                else:
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pl:
                        pl_page = pl.pages[page_no - 1]
                        cropped = pl_page.within_bbox((bbox[0], bbox[1], bbox[2], bbox[3]))
                        content = cropped.extract_text() or ""
            else:
                # Run OCR crop
                pdf_w = doc[page_no - 1].rect.width
                pdf_h = doc[page_no - 1].rect.height
                page_img = Image.open(page_images[page_no - 1])
                img_w, img_h = page_img.size
                
                crop_box = [
                    bbox[0] * (img_w / pdf_w),
                    bbox[1] * (img_h / pdf_h),
                    bbox[2] * (img_w / pdf_w),
                    bbox[3] * (img_h / pdf_h)
                ]
                cropped_img = page_img.crop(crop_box)
                
                if cfg["ocr"] == "tesseract":
                    from algorithms.text_extraction.scanned.tesseract.extractor import extract_text as tesseract_ocr
                    res = tesseract_ocr(cropped_img)
                else:
                    from algorithms.text_extraction.scanned.easyocr.extractor import extract_text as easy_ocr
                    res = easy_ocr(cropped_img)
                content = res.get("full_text", "").strip()

                
        element["content"] = content
        final_elements.append(element)

    # 5.5 Post-processing and Mapping (Steps 2, 3, 4, 5)
    print("\n[POST-PROCESSING] Cleaning text, assigning IDs, mapping types and tables/figures...")
    import re
    # Load manifest info to get collision-safe doc_id
    manifest_filename = os.path.join(doc_output_dir, "manifest.json")
    with open(manifest_filename, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    doc_id = manifest["doc_id"]

    
    def get_bbox_overlap(box1, box2):
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - intersection_area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def map_element_type(act_type, content):
        mapping = {
            "title": "title",
            "plain text": "paragraph",
            "figure": "figure",
            "figure_caption": "paragraph",
            "table": "table",
            "table_caption": "paragraph",
            "table_footnote": "paragraph",
            "isolate_formula": "formula",
            "formula_caption": "paragraph",
            "abandon": "boilerplate"
        }
        mapped = mapping.get(act_type, "paragraph")
        
        # Heuristic to detect headings/appendix titles in captions/text
        if mapped != "title" and content:
            cleaned_text = content.strip()
            is_section_header = False
            # E.g. "E\nTraining corpus", "E Training corpus"
            if re.match(r'^[A-Z]\s*\n?\s+[a-zA-Z]', cleaned_text):
                is_section_header = True
            elif re.match(r'^Appendix\s+[A-Z]', cleaned_text):
                is_section_header = True
            
            if is_section_header:
                return "title"
                
        return mapped

    def get_heading_level_and_clean_text(text, last_level):
        t_clean = text.strip()
        match = re.match(r'^([0-9a-zA-Z]+(?:\.[0-9a-zA-Z]+)*)\s+(.*)', t_clean)
        if match:
            prefix, rest = match.groups()
            parts = prefix.split('.')
            parts = [p for p in parts if p.strip()]
            is_prefix = False
            if len(parts) > 1:
                is_prefix = True
            elif len(parts) == 1:
                p = parts[0]
                if p.isdigit() or (p.isupper() and len(p) <= 3):
                    is_prefix = True
            if is_prefix:
                return len(parts), prefix, rest.strip()
        return last_level, "", t_clean

    def sort_page_reading_order(page_elements):
        # Sort page elements by y0 first to get initial top-to-bottom order
        sorted_by_y0 = sorted(page_elements, key=lambda el: el["bbox"][1])
        
        slices = []
        current_slice = []
        for el in sorted_by_y0:
            x0, y0, x1, y1 = el["bbox"]
            is_full_width = (x0 < 240 and x1 > 350)
            if is_full_width:
                if current_slice:
                    slices.append((False, current_slice))
                    current_slice = []
                slices.append((True, [el]))
            else:
                current_slice.append(el)
        if current_slice:
            slices.append((False, current_slice))
            
        final_order = []
        for is_full, slice_els in slices:
            if is_full:
                final_order.extend(slice_els)
            else:
                left_col = []
                right_col = []
                for el in slice_els:
                    cx = (el["bbox"][0] + el["bbox"][2]) / 2
                    if cx < 295:
                        left_col.append(el)
                    else:
                        right_col.append(el)
                left_col = sorted(left_col, key=lambda e: e["bbox"][1])
                right_col = sorted(right_col, key=lambda e: e["bbox"][1])
                final_order.extend(left_col)
                final_order.extend(right_col)
        return final_order

    # Step 4: Hyphenation OCR artifact cleanup on all contents
    for el in final_elements:
        if "content" in el and el["content"]:
            el["content"] = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', el["content"])

    # Remap table_markdown
    for el in final_elements:
        if el["type"] == "table":
            val = ""
            if "extracted" in el and isinstance(el["extracted"], dict):
                val = el["extracted"].get("markdown", "") or ""
            if not val.strip():
                val = "| Table |\n| --- |\n| (No tabular data extracted) |"
            el["table_markdown"] = val
            if el["table_markdown"]:
                el["table_markdown"] = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', el["table_markdown"])

    # Remap figure image_caption and image_path
    for el in final_elements:
        if el["type"] == "figure":
            best_caption = ""
            best_img_path = ""
            for fig in extracted_figures:
                if fig["page"] == el["page"]:
                    overlap = get_bbox_overlap(el["bbox"], fig["bbox"])
                    if overlap > 0.1:
                        best_caption = fig["caption"] or ""
                        best_img_path = fig.get("image_path", "") or ""
                        break
            if not best_caption.strip():
                best_caption = "Visual representation from the document."
            if not best_img_path or not os.path.exists(best_img_path):
                # Fallback to page image frame
                page_img_path = page_images[el["page"] - 1]
                best_img_path = page_img_path
            el["image_caption"] = best_caption
            el["image_path"] = best_img_path

    # Assign reading_order per page (column-aware)
    elements_by_page = {}
    for el in final_elements:
        elements_by_page.setdefault(el["page"], []).append(el)
        
    sorted_final_elements = []
    for page_no in sorted(elements_by_page.keys()):
        page_els = elements_by_page[page_no]
        sorted_page_els = sort_page_reading_order(page_els)
        for r_idx, el in enumerate(sorted_page_els):
            el["reading_order"] = r_idx
        sorted_final_elements.extend(sorted_page_els)
    final_elements = sorted_final_elements

    # Assign element_id, page_number, element_type, and text
    for idx, el in enumerate(final_elements):
        el["element_id"] = f"{doc_id}_{idx:04d}"
        el["page_number"] = el["page"]
        el["element_type"] = map_element_type(el["type"], el.get("content"))
        
        # Populate text
        if el["element_type"] in ["title", "paragraph", "header", "footer", "list_item", "formula"]:
            el["text"] = el.get("content", "")
        else:
            el["text"] = None

        # Step 2: Normalize whitespace (replace \n and repeated spaces with a single space) in title text
        if el["element_type"] == "title":
            if el["text"]:
                el["text"] = re.sub(r'\s+', ' ', el["text"]).strip()
            if el["content"]:
                el["content"] = re.sub(r'\s+', ' ', el["content"]).strip()

    # Step 5: Construct section_path hierarchy state machine
    stack = []
    last_level = 1
    for el in final_elements:
        if el["element_type"] == "title":
            text = el.get("text", "") or ""
            level, prefix, clean_text = get_heading_level_and_clean_text(text, last_level)
            while len(stack) >= level:
                stack.pop()
            stack.append(text)
            el["section_path"] = list(stack)
            last_level = level
        else:
            el["section_path"] = list(stack)

    # Ensure all elements have the exact same set of 15 keys for schema consistency (Check 10)
    all_keys = [
        "type", "bbox", "confidence", "page", "content", "extracted",
        "table_markdown", "image_caption", "image_path", "reading_order",
        "element_id", "page_number", "element_type", "text", "section_path"
    ]
    for el in final_elements:
        # Clean replacement characters in string fields
        if "content" in el and isinstance(el["content"], str):
            el["content"] = el["content"].replace("\ufffd", "")
        if "text" in el and isinstance(el["text"], str):
            el["text"] = el["text"].replace("\ufffd", "")
        if "table_markdown" in el and isinstance(el["table_markdown"], str):
            el["table_markdown"] = el["table_markdown"].replace("\ufffd", "")
        if "image_caption" in el and isinstance(el["image_caption"], str):
            el["image_caption"] = el["image_caption"].replace("\ufffd", "")
            
        for k in all_keys:
            if k not in el:
                el[k] = None

    # Print section_path for Scientific_001 titles as sanity check
    if doc_id == "Scientific_001":
        print("\n===== SANITY CHECK: Section Paths for Scientific_001 titles =====")
        for el in final_elements:
            if el["element_type"] == "title":
                print(f"  Title: '{el.get('text', '').strip()}' -> Path: {el.get('section_path')}")
        print("=================================================================\n")



    # 6. Save Outputs
    print("\n[STEP 6] Packaging structured JSON & annotated PDF...")
    
    # Filter boilerplate elements out of main elements list into discarded_elements
    filtered_elements = [el for el in final_elements if el["element_type"] != "boilerplate"]
    discarded_elements = [el for el in final_elements if el["element_type"] == "boilerplate"]
    print(f"  [Boilerplate Filter] Document {doc_id} elements before: {len(final_elements)}")
    print(f"  [Boilerplate Filter] Elements kept: {len(filtered_elements)} | Discarded: {len(discarded_elements)}")

    document_json = {
        "doc_id": doc_id,
        "content_hash": manifest["content_hash"],
        "source_filename": manifest["source_filename"],
        "source_path": manifest["source_path"],
        "access_tags": manifest["access_tags"],
        "ingested_at": manifest["ingested_at"],
        "file_modified_at": manifest["file_modified_at"],
        "metadata": {
            "filename": os.path.basename(pdf_path),
            "page_count": page_count,
            "pipeline_config": cfg,
            "scanned": not has_digital_text,
            "elapsed_seconds": round(time.time() - start_time, 2)
        },
        "elements": filtered_elements,
        "discarded_elements": discarded_elements,



        "tables": [{
            "table_index": t["table_index"],
            "page": t["page"],
            "markdown": t["markdown"],
            "engine": t["engine"]
        } for t in extracted_tables],
        "visual_captions": extracted_figures
    }
    
    json_path = os.path.join(doc_output_dir, f"{doc_stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(document_json, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Saved structured JSON to: {json_path}")

    # Generate annotated PDF
    anno_pdf = fitz.open(pdf_path)
    for element in final_elements:
        page_no = element["page"]
        el_type = element["type"]
        bbox = element["bbox"]
        
        color = COLOR_MAP.get(el_type, DEFAULT_COLOR)
        page_obj = anno_pdf[page_no - 1]
        
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        if rect.is_valid and not rect.is_empty:
            page_obj.draw_rect(rect, color=color, width=1.5)
            # draw small text tag
            tag_rect = fitz.Rect(bbox[0], bbox[1] - 8, bbox[0] + 80, bbox[1])
            page_obj.draw_rect(tag_rect, color=color, fill=color)
            page_obj.insert_text(fitz.Point(bbox[0] + 2, bbox[1] - 2), el_type, fontsize=6, color=(1, 1, 1))
            
    bbox_pdf_path = os.path.join(doc_output_dir, f"{doc_stem}_bbox.pdf")
    anno_pdf.save(bbox_pdf_path)
    anno_pdf.close()
    print(f"  [OK] Saved annotated BBox PDF to: {bbox_pdf_path}")
    
    total_time = time.time() - start_time
    print(f"\n[OK] Pipeline completed in {total_time:.2f} seconds!")
    print("=" * 60)
    return json_path

# ── Command Line Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Master Document Understanding End-to-End Extraction Pipeline CLI Orchestrator."
    )
    parser.add_argument("--pdf", required=True, help="Path to input PDF file to process.")
    parser.add_argument("--layout", choices=["doclayout_yolo", "nemotron_parse", "landingai_ade"], help="Layout extraction option.")
    parser.add_argument("--text", choices=["pymupdf", "pdfplumber"], help="Native text extraction option.")
    parser.add_argument("--ocr", choices=["tesseract", "easyocr"], help="Scanned text OCR option.")
    parser.add_argument("--table", choices=["docling_tableformer", "tatr"], help="Table extraction option.")
    parser.add_argument("--image", choices=["groq_llama", "groq_qwen", "local_qwen", "local_moondream"], help="Embedded image extraction description option.")
    parser.add_argument("--output", default=OUTPUT_DIR, help=f"Directory to save outputs (default: {OUTPUT_DIR}).")

    
    args = parser.parse_args()
    
    overrides = {
        "layout_detection": args.layout,
        "text_extraction": args.text,
        "ocr": args.ocr,
        "table_extraction": args.table,
        "image_extraction": args.image
    }
    
    if not os.path.exists(args.pdf):
        print(f"[ERROR] Input PDF not found: {args.pdf}")
        sys.exit(1)
        
    try:
        run_pipeline(args.pdf, args.output, overrides)
    except Exception as e:
        print(f"[CRITICAL] Pipeline crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
