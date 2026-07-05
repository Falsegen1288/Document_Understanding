import uuid
import os
import logging
import threading
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
import fitz  # PyMuPDF

from backend.core.config import settings
from backend.core.database import get_db, Job
from backend.core.storage import save_uploaded_pdf
from backend.api.schemas import JobCreateResponse

logger = logging.getLogger("upload_route")

router = APIRouter()

def is_redis_available() -> bool:
    """Check if Redis connection is active and responsive."""
    try:
        import redis
        r = redis.Redis.from_url(settings.REDIS_URL, socket_timeout=1.0)
        return r.ping()
    except Exception:
        return False

def trigger_background_pipeline(job_id: str, background_tasks: BackgroundTasks):
    """
    Triggers job execution asynchronously. Uses Celery if Redis is available,
    otherwise falls back gracefully to a native background thread/FastAPI BackgroundTasks.
    """
    if is_redis_available():
        try:
            from backend.tasks.pipeline import process_pdf_job
            process_pdf_job.delay(job_id)
            logger.info(f"Queued job {job_id} using Celery worker.")
            return "queued (celery)"
        except Exception as e:
            logger.warning(f"Failed to queue job {job_id} on Celery: {e}. Falling back to BackgroundTasks.")
    
    # Fallback to local background thread execution
    from backend.services.pipeline_runner import run_pipeline_job
    
    # Run using native python threading so that the FastAPI request thread is not blocked
    thread = threading.Thread(target=run_pipeline_job, args=(job_id,), daemon=True)
    thread.start()
    logger.info(f"Triggered job {job_id} using native background thread fallback.")
    return "queued (background_thread)"

@router.post("/upload", response_model=JobCreateResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form("auto"),
    layout_algo: str = Form("doclayout_yolo"),
    ocr_algo: str = Form("easyocr"),
    table_algo: str = Form("tatr"),
    figure_algo: str = Form("groq_llama"),
    db: Session = Depends(get_db)

):
    # 1. Size Validation (50MB limit)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum allowed size of 50MB. (Uploaded size: {file_size / (1024*1024):.1f}MB)"
        )
    
    # 2. Magic Bytes Validation (Check for %PDF)
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Uploaded file is not a valid PDF document."
        )
        
    # Generate unique Job ID
    job_id = str(uuid.uuid4())
    
    # Save the PDF file to storage/uploads/{job_id}.pdf
    try:
        pdf_path = save_uploaded_pdf(content, job_id)
    except Exception as e:
        logger.error(f"Failed to save uploaded PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded document on the server.")
    
    # 3. Scan for Page Count before queuing (PyMuPDF)
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception as e:
        # Cleanup uploaded file on error
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse PDF document. It may be corrupted. Error: {e}"
        )
        
    if page_count > 200:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        raise HTTPException(
            status_code=400,
            detail=f"PDF document exceeds the maximum limit of 200 pages (File contains {page_count} pages)."
        )
        
    # 4. Save metadata into SQLite database
    new_job = Job(
        id=job_id,
        filename=file.filename,
        doc_type=doc_type,
        status="queued",
        layout_algo=layout_algo,
        ocr_algo=ocr_algo,
        table_algo=table_algo,
        figure_algo=figure_algo,
        total_pages=page_count,
        pages_done=0
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    
    # 5. Trigger async processing (Celery vs Local Thread)
    queue_status = trigger_background_pipeline(job_id, background_tasks)
    
    return JobCreateResponse(
        job_id=job_id,
        status="queued"
    )
