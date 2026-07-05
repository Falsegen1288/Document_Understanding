import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from backend.core.storage import (
    get_result_json_path,
    get_page_result_json_path,
    get_page_image_path,
    get_pdf_path,
    get_job_results_dir
)
import fitz

logger = logging.getLogger("results_route")

router = APIRouter()

@router.get("/results/{job_id}")
def get_full_results(job_id: str):
    """Retrieve full merged result.json for a completed job."""
    json_path = get_result_json_path(job_id)
    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Result JSON for job '{job_id}' not found. Job may be still running or failed."
        )
    return FileResponse(json_path, media_type="application/json", filename="result.json")

@router.get("/results/{job_id}/page/{page_num}")
def get_page_results(job_id: str, page_num: int):
    """Retrieve single page JSON result.json for a job."""
    json_path = get_page_result_json_path(job_id, page_num)
    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Page {page_num} JSON result for job '{job_id}' not found."
        )
    return FileResponse(json_path, media_type="application/json", filename=f"page_{page_num}.json")

@router.get("/results/{job_id}/bbox-pdf")
def download_bbox_pdf(job_id: str):
    """Download the annotated PDF file with bounding boxes drawn."""
    results_dir = get_job_results_dir(job_id)
    bbox_pdf_path = os.path.join(results_dir, f"{job_id}_bbox.pdf")
    
    if not os.path.exists(bbox_pdf_path):
        # Fallback to original PDF if bbox PDF hasn't finished rendering
        pdf_path = get_pdf_path(job_id)
        if os.path.exists(pdf_path):
            return FileResponse(pdf_path, media_type="application/pdf", filename=f"document_{job_id}.pdf")
        raise HTTPException(status_code=404, detail="Document PDF file not found.")
        
    return FileResponse(bbox_pdf_path, media_type="application/pdf", filename=f"annotated_{job_id}.pdf")

@router.get("/pages/{job_id}/{page_num}")
def get_page_image(job_id: str, page_num: int):
    """
    Get the PNG image of a page. Renders the page on-demand at 200 DPI
    if it was not pre-rendered during pipeline execution.
    """
    img_path = get_page_image_path(job_id, page_num)
    
    # Render on demand if file does not exist yet
    if not os.path.exists(img_path):
        pdf_path = get_pdf_path(job_id)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Source PDF document not found.")
            
        try:
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                raise HTTPException(status_code=400, detail=f"Page number {page_num} is out of range.")
                
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=200)
            
            # Ensure folder exists
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            pix.save(img_path)
            doc.close()
            logger.info(f"Rendered page {page_num} on-demand for job {job_id}.")
        except Exception as e:
            logger.error(f"On-demand page rendering failed: {e}")
            raise HTTPException(status_code=500, detail=f"On-demand page rendering failed: {e}")

    return FileResponse(img_path, media_type="image/png")
