import os
import sys
import time
import json
import traceback
import logging
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd

# Add root folders to sys.path
SERVICES_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_ROOT = os.path.abspath(os.path.join(SERVICES_DIR, "..", "..")) # obsidian-precision
WORKSPACE_ROOT = os.path.abspath(os.path.join(PARENT_ROOT, ".."))      # Document_Understanding

for path in [PARENT_ROOT, WORKSPACE_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.core.config import settings
from backend.core.database import SessionLocal, Job
from backend.core.storage import (
    get_pdf_path,
    get_page_image_path,
    save_page_result_json,
    save_result_json,
    get_result_json_path,
    get_job_pages_dir,
    get_job_results_dir
)
from backend.services.sse_manager import sse_manager

# Re-use your polished main.py helpers directly!
from main import (
    compile_grounded_context_and_prompt,
    compute_semantic_similarity,
    critique_and_rewrite_caption,
    _pre_extract_page_text,
    merge_tables_into_elements,
    COLOR_MAP,
    DEFAULT_COLOR
)

logger = logging.getLogger("pipeline_runner")

def update_job_status(job_id: str, status: str, pages_done: int = None, total_pages: int = None, error_message: str = None, result_path: str = None):
    """Utility to update a job's state in SQLite database thread-safely."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            if pages_done is not None:
                job.pages_done = pages_done
            if total_pages is not None:
                job.total_pages = total_pages
            if error_message is not None:
                job.error_message = error_message
            if result_path is not None:
                job.result_path = result_path
            if status in ["done", "failed"]:
                job.completed_at = datetime.utcnow()
            db.commit()
            logger.info(f"Updated job {job_id} to status: {status} (page {pages_done}/{total_pages})")
    except Exception as db_err:
        logger.error(f"Database update failed for job {job_id}: {db_err}")
    finally:
        db.close()

def run_pipeline_job(job_id: str):
    """
    Main pipeline task executor. Runs end-to-end extraction page-by-page,
    updating the database and sending real-time progress events via SSE.
    """
    start_time = time.time()
    logger.info(f"Starting pipeline runner for job {job_id}...")
    
    # 1. Fetch job metadata
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        logger.error(f"Job {job_id} not found in database.")
        db.close()
        return
    
    doc_type = job.doc_type
    layout_algo = job.layout_algo
    ocr_algo = job.ocr_algo
    table_algo = job.table_algo
    figure_algo = job.figure_algo
    filename = job.filename
    db.close()
    
    # Configure VLM API Key environment variables for pipeline subprocesses
    # We read settings keys and map them back to standard pipeline env vars
    os.environ["GROQ_API_KEY"] = settings.GROQ_API_KEY
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    
    pdf_path = get_pdf_path(job_id)
    if not os.path.exists(pdf_path):
        err = f"PDF file not found at expected path: {pdf_path}"
        logger.error(err)
        update_job_status(job_id, "failed", error_message=err)
        sse_manager.publish_threadsafe(job_id, "job_failed", {"error": err})
        return

    update_job_status(job_id, "running")
    sse_manager.publish_threadsafe(job_id, "job_start", {"filename": filename})

    doc = None
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        update_job_status(job_id, "running", pages_done=0, total_pages=total_pages)
        
        # Check for digitally embedded text
        has_digital_text = False
        for page in doc:
            if page.get_text("text").strip():
                has_digital_text = True
                break
        logger.info(f"Job {job_id} native digital text: {has_digital_text}")

        # Config setup for algorithms
        cfg = {
            "layout_detection": layout_algo,
            "text_extraction": "pymupdf",
            "ocr": ocr_algo,
            "table_extraction": table_algo,
            "image_extraction": figure_algo
        }

        # Lazy load algorithm runners based on selection to conserve memory
        layout_detect = None
        tf_extract = None
        tatr_extract = None

        if layout_algo == "doclayout_yolo":
            from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as layout_detect
        elif layout_algo == "nemotron_parse":
            from algorithms.layout_detection.nemotron_parse.extractor import detect_layout as layout_detect
        elif layout_algo == "landingai_ade":
            from algorithms.layout_detection.landingai_ade.extractor import detect_layout as layout_detect
        else:
            from algorithms.layout_detection.doclayout_yolo.extractor import detect_layout as layout_detect

        if table_algo == "docling_tableformer":
            from algorithms.table_extraction.docling_tableformer.extractor import extract_tables as tf_extract
        else:
            from algorithms.table_extraction.tatr.extractor import extract_tables as tatr_extract


        # Maintain list of page result summaries
        pages_results = []
        succeeded_pages = 0

        # Step 2: Progressive page-by-page loop
        for p_idx in range(total_pages):
            page_num = p_idx + 1
            logger.info(f"Processing job {job_id} | Page {page_num}/{total_pages}")
            sse_manager.publish_threadsafe(job_id, "page_start", {"page": page_num, "total": total_pages})
            
            try:
                # 2.1 Render Page PNG at 200 DPI
                page_obj = doc[p_idx]
                pix = page_obj.get_pixmap(dpi=200)
                img_path = get_page_image_path(job_id, page_num)
                pix.save(img_path)
                
                img = Image.open(img_path)
                img_w, img_h = img.size
                pdf_w = page_obj.rect.width
                pdf_h = page_obj.rect.height
                scale_x = img_w / pdf_w
                scale_y = img_h / pdf_h

                # 2.2 Run Layout Detection
                sse_manager.publish_threadsafe(job_id, "layout_start", {"page": page_num})
                detected_elements = []
                
                if layout_detect:
                    res = layout_detect(img)
                    for det_idx, r in enumerate(res):
                        detected_elements.append({
                            "id": f"det_{page_num}_{det_idx:03d}",
                            "label": r["type"],
                            "type": r["type"],
                            "class_id": r.get("class_id", 0),
                            "confidence": round(r.get("confidence", 0.90), 2),
                            "page": page_num,
                            "bbox": [
                                r["bbox"][0] / scale_x,
                                r["bbox"][1] / scale_y,
                                r["bbox"][2] / scale_x,
                                r["bbox"][3] / scale_y
                            ],
                            "content": ""
                        })

                
                logger.info(f"Page {page_num} layout bboxes: {len(detected_elements)}")
                sse_manager.publish_threadsafe(job_id, "layout_done", {"page": page_num, "detections": detected_elements})

                # 2.3 Context Pre-extraction (Page text block OCR)
                # Ensure text is pre-extracted for captions
                _pre_extract_page_text(page_num, detected_elements, doc, pdf_path, cfg, has_digital_text, [img_path]*total_pages)

                # 2.4 Table Grid Extraction
                sse_manager.publish_threadsafe(job_id, "table_start", {"page": page_num})
                page_tables = []
                table_bboxes = [el for el in detected_elements if el["label"] == "table"]
                
                if table_bboxes:
                    logger.info(f"Page {page_num} contains table bounding boxes. Running table extraction...")
                    if table_algo == "docling_tableformer" and tf_extract:
                        page_tables = tf_extract(pdf_path, pages=[page_num])
                    elif tatr_extract:
                        # Crop table image crops from 200 DPI image
                        for t_idx, t_bbox in enumerate(table_bboxes):
                            bbox = t_bbox["bbox"]
                            crop_box = [
                                bbox[0] * scale_x,
                                bbox[1] * scale_y,
                                bbox[2] * scale_x,
                                bbox[3] * scale_y
                            ]
                            cropped_table = img.crop(crop_box)
                            tatr_res = tatr_extract(cropped_table)
                            for tr in tatr_res:
                                tr["page"] = page_num
                                tr["table_index"] = len(page_tables) + 1
                                page_tables.append(tr)

                
                logger.info(f"Page {page_num} tables parsed: {len(page_tables)}")
                sse_manager.publish_threadsafe(job_id, "table_done", {"page": page_num, "tables_found": len(page_tables)})

                # Merge extracted tables into detected elements so VLG engine can associate them
                merge_tables_into_elements(page_num, page_tables, detected_elements)

                # 2.5 Figures Visual Grounded Captioning
                sse_manager.publish_threadsafe(job_id, "figures_start", {"page": page_num})
                figure_bboxes = [el for el in detected_elements if el["label"] in ["figure", "image"]]
                figures_found = len(figure_bboxes)
                
                for f_idx, figure_el in enumerate(figure_bboxes):
                    bbox = figure_el["bbox"]
                    
                    # 1. Crop figure at high DPI
                    crop_box = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                    cropped_fig = img.crop(crop_box)
                    crop_filename = f"page_{page_num}_fig_{f_idx}.png"
                    crop_dir = get_job_pages_dir(job_id)
                    crop_path = os.path.join(crop_dir, crop_filename)
                    cropped_fig.save(crop_path)
                    
                    # 2. Compile grounded context and prompt
                    grounded_prompt, context_text, heading, domain, matched_table = compile_grounded_context_and_prompt(
                        page_num, bbox, detected_elements, pdf_path
                    )
                    
                    table_crop_url = ""
                    if matched_table:
                        try:
                            t_bbox = matched_table["bbox"]
                            t_crop_box = [
                                t_bbox[0] * scale_x,
                                t_bbox[1] * scale_y,
                                t_bbox[2] * scale_x,
                                t_bbox[3] * scale_y
                            ]
                            cropped_t = img.crop(t_crop_box)
                            t_crop_filename = f"page_{page_num}_fig_{f_idx}_table.png"
                            t_crop_path = os.path.join(crop_dir, t_crop_filename)
                            cropped_t.save(t_crop_path)
                            table_crop_url = f"storage/pages/{job_id}/{t_crop_filename}"
                        except Exception as t_err:
                            logger.warning(f"Table cropping failed: {t_err}")
                    
                    desc_text = ""
                    # 3. Call VLM model with proper key checking
                    try:
                        if figure_algo == "groq_llama":
                            from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
                            res = groq_desc(cropped_fig, prompt=grounded_prompt, api_key=settings.GROQ_API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")
                            desc_text = res.get("description", "")
                        elif figure_algo == "groq_qwen":
                            from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
                            res = groq_desc(cropped_fig, prompt=grounded_prompt, api_key=settings.GROQ_API_KEY, model="qwen/qwen3.6-27b")
                            desc_text = res.get("description", "")
                        elif figure_algo == "local_qwen":
                            from algorithms.image_extraction.local.extractor import describe_figure as local_desc
                            res = local_desc(cropped_fig, prompt=grounded_prompt, model="qwen2.5vl:3b")
                            desc_text = res.get("description", "")
                        elif figure_algo == "local_moondream":
                            from algorithms.image_extraction.local.extractor import describe_figure as local_desc
                            res = local_desc(cropped_fig, prompt=grounded_prompt, model="moondream:latest")
                            desc_text = res.get("description", "")
                        else:
                            from algorithms.image_extraction.groq.extractor import describe_figure as groq_desc
                            res = groq_desc(cropped_fig, prompt=grounded_prompt, api_key=settings.GROQ_API_KEY)
                            desc_text = res.get("description", "")

                    except Exception as vlm_err:
                        logger.warning(f"VLM visual captioning failed: {vlm_err}")
                        desc_text = f"Figure crop showing layout structure. VLM descriptor failed: {vlm_err}"

                    # 4. Neural similarity scoring
                    sim_score = 0.0
                    try:
                        sim_score = compute_semantic_similarity(desc_text, context_text)
                        
                        # Critique loop
                        if sim_score < 0.20 and context_text.strip():
                            desc_text = critique_and_rewrite_caption(cropped_fig, desc_text, context_text, figure_algo, api_key=os.environ.get(f"{figure_algo.upper()}_API_KEY"))
                            sim_score = compute_semantic_similarity(desc_text, context_text)
                    except Exception as sim_err:
                        logger.warning(f"Semantic similarity scoring skipped/failed: {sim_err}")
                    
                    # Store grounded VLM figure description
                    figure_el["extracted"] = {
                        "type": "figure",
                        "caption_ocr": f"Figure detection page {page_num}",
                        "vlm_description": desc_text,
                        "vlm_model": f"{figure_algo} vision-api",
                        "prompt_schema": domain,
                        "similarity_score": round(sim_score, 3),
                        "nearest_heading": heading,
                        "crop_path": f"storage/pages/{job_id}/{crop_filename}",
                        "grounded_context": context_text,
                        "table_crop_path": table_crop_url
                    }
                    
                sse_manager.publish_threadsafe(job_id, "figures_done", {"page": page_num, "figures_found": figures_found})

                # 2.6 Route remaining text blocks and OCR
                sse_manager.publish_threadsafe(job_id, "ocr_start", {"page": page_num})
                for element in detected_elements:
                    el_type = element["label"]
                    bbox = element["bbox"]
                    
                    if el_type in ["table", "figure", "image"]:
                        continue
                        
                    content = element.get("content", "").strip()
                    
                    if not content:
                        if has_digital_text:
                            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                            content = page_obj.get_text("text", clip=rect).strip()
                        else:
                            # Run Scanned OCR
                            crop_box = [
                                bbox[0] * scale_x,
                                bbox[1] * scale_y,
                                bbox[2] * scale_x,
                                bbox[3] * scale_y
                            ]
                            cropped_text = img.crop(crop_box)
                            try:
                                if ocr_algo == "tesseract":
                                    from algorithms.text_extraction.scanned.tesseract.extractor import extract_text as tesseract_ocr
                                    ocr_res = tesseract_ocr(cropped_text)
                                else:
                                    from algorithms.text_extraction.scanned.easyocr.extractor import extract_text as easy_ocr
                                    ocr_res = easy_ocr(cropped_text)
                                content = ocr_res.get("full_text", "").strip()

                            except Exception as ocr_err:
                                logger.error(f"OCR region extraction failed: {ocr_err}")
                                content = ""
                    
                    element["extracted"] = {
                        "type": "text",
                        "content": content,
                        "ocr_engine": ocr_algo if not has_digital_text else "pymupdf"
                    }

                # 2.7 Merge Table Results into BBox detections
                for table in page_tables:
                    if table.get("matched"):
                        continue
                    # Match table item to a table bbox on page if possible (nearest distance)
                    # For simplicity, if TATR is used, bboxes match perfectly.
                    # If Docling is used, we can find the closest "table" labeled bbox to the docling table coordinates,
                    # or append a new table detection!
                    # Let's map table records inside the detections list
                    table_item = {
                        "id": f"table_{page_num}_{table.get('table_index', 0)}",
                        "label": "table",
                        "type": "table",
                        "class_id": 5,
                        "confidence": 0.90,
                        "page": page_num,
                        "bbox": table.get("bbox", [0, 0, pdf_w, pdf_h]), # fallback to full page if missing
                        "extracted": {
                            "type": "table",
                            "markdown": table.get("markdown", ""),
                            "dataframe_csv": pd.DataFrame(table.get("data", [])).to_csv(index=False) if table.get("data") else "",
                            "extractor": table.get("engine", "tatr"),
                            "rows": len(table.get("data", [])),
                            "cols": len(table.get("data", [[]])[0]) if table.get("data") else 0
                        }
                    }
                    detected_elements.append(table_item)

                sse_manager.publish_threadsafe(job_id, "ocr_done", {"page": page_num})

                # 2.8 Save progressive page result JSON
                page_result = {
                    "page_number": page_num,
                    "page_image_path": f"storage/pages/{job_id}/page_{page_num}.png",
                    "page_width_px": pdf_w,
                    "page_height_px": pdf_h,
                    "detections": detected_elements
                }
                
                save_page_result_json(job_id, page_num, page_result)
                pages_results.append(page_result)
                succeeded_pages += 1
                
                update_job_status(job_id, "running", pages_done=page_num, total_pages=total_pages)
                sse_manager.publish_threadsafe(job_id, "page_complete", {
                    "page": page_num,
                    "total": total_pages,
                    "result": page_result
                })

            except Exception as page_ex:
                logger.error(f"Error processing page {page_num} for job {job_id}: {page_ex}")
                logger.error(traceback.format_exc())
                
                # Graceful page fallback (mark page status as failed but don't crash job)
                failed_page_result = {
                    "page_number": page_num,
                    "page_image_path": f"storage/pages/{job_id}/page_{page_num}.png",
                    "page_width_px": 0,
                    "page_height_px": 0,
                    "detections": [],
                    "status": "failed",
                    "error_message": str(page_ex)
                }
                save_page_result_json(job_id, page_num, failed_page_result)
                pages_results.append(failed_page_result)
                
                update_job_status(job_id, "running", pages_done=page_num, total_pages=total_pages)
                sse_manager.publish_threadsafe(job_id, "page_complete", {
                    "page": page_num,
                    "total": total_pages,
                    "result": failed_page_result
                })

        # Step 3: Complete execution & generate final merge
        logger.info(f"Completing pipeline for job {job_id}. Successful pages: {succeeded_pages}/{total_pages}")
        
        # Cleanup cached ODL results if any
        if f"_odl_cache_{job_id}" in globals():
            del globals()[f"_odl_cache_{job_id}"]
            
        full_json = {
            "job_id": job_id,
            "filename": filename,
            "doc_type": doc_type,
            "pipeline": cfg,
            "pages": pages_results
        }
        
        results_dir = get_job_results_dir(job_id)
        result_path = save_result_json(job_id, full_json)
        
        # 3.1 Generate annotated PDF with bounding boxes
        try:
            anno_pdf = fitz.open(pdf_path)
            for page_res in pages_results:
                p_num = page_res["page_number"]
                if page_res.get("status") == "failed":
                    continue
                p_obj = anno_pdf[p_num - 1]
                
                for det in page_res["detections"]:
                    el_type = det["label"]
                    bbox = det["bbox"]
                    color = COLOR_MAP.get(el_type, DEFAULT_COLOR)
                    
                    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                    if rect.is_valid and not rect.is_empty:
                        p_obj.draw_rect(rect, color=color, width=1.5)
                        tag_rect = fitz.Rect(bbox[0], bbox[1] - 8, bbox[0] + 60, bbox[1])
                        p_obj.draw_rect(tag_rect, color=color, fill=color)
                        p_obj.insert_text(fitz.Point(bbox[0] + 2, bbox[1] - 2), el_type, fontsize=6, color=(1, 1, 1))
            
            bbox_pdf_output = os.path.join(results_dir, f"{job_id}_bbox.pdf")
            anno_pdf.save(bbox_pdf_output)
            anno_pdf.close()
            logger.info(f"Saved annotated bbox PDF to {bbox_pdf_output}")
        except Exception as anno_err:
            logger.warning(f"Failed to generate annotated PDF: {anno_err}")

        # Update database with completion state
        update_job_status(job_id, "done", pages_done=total_pages, result_path=result_path)
        elapsed = round(time.time() - start_time, 2)
        sse_manager.publish_threadsafe(job_id, "job_complete", {
            "elapsed_seconds": elapsed,
            "succeeded_pages": succeeded_pages,
            "total_pages": total_pages,
            "result_path": result_path
        })
        
    except Exception as job_ex:
        logger.error(f"Critical pipeline failure for job {job_id}: {job_ex}")
        logger.error(traceback.format_exc())
        update_job_status(job_id, "failed", error_message=str(job_ex))
        sse_manager.publish_threadsafe(job_id, "job_failed", {"error": str(job_ex)})
    finally:
        if doc:
            doc.close()
