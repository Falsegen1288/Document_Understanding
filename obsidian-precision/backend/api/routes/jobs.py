import os
import shutil
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from backend.core.database import get_db, Job
from backend.core.config import settings
from backend.core.storage import get_pdf_path, get_job_pages_dir, get_job_results_dir
from backend.api.schemas import JobStatusResponse

logger = logging.getLogger("jobs_route")

router = APIRouter()

@router.get("/jobs/history", response_model=List[JobStatusResponse])
def get_job_history(db: Session = Depends(get_db)):
    """Fetch history of the last 20 jobs from SQLite, ordered by created_at descending."""
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(20).all()
    return [job.to_dict() for job in jobs]

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Fetch status and progress for a single job."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")
    return job.to_dict()

@router.delete("/jobs/{job_id}")
def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a job from SQLite and permanently remove all its generated files from storage."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")
    
    # 1. Clean up physical files
    # Uploaded PDF
    pdf_path = get_pdf_path(job_id)
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as e:
            logger.warning(f"Failed to delete raw PDF {pdf_path}: {e}")
            
    # Generated Page PNG crops
    pages_dir = get_job_pages_dir(job_id)
    if os.path.exists(pages_dir):
        try:
            shutil.rmtree(pages_dir)
        except Exception as e:
            logger.warning(f"Failed to delete page PNGs directory {pages_dir}: {e}")
            
    # JSON results and annotated PDF
    results_dir = get_job_results_dir(job_id)
    if os.path.exists(results_dir):
        try:
            shutil.rmtree(results_dir)
        except Exception as e:
            logger.warning(f"Failed to delete results directory {results_dir}: {e}")

    # 2. Delete database entry
    db.delete(job)
    db.commit()
    
    logger.info(f"Permanently deleted job {job_id} and all related storage files.")
    return {"deleted": True, "job_id": job_id}
