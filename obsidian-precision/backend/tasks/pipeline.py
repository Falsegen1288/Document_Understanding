import logging
from backend.tasks.celery_app import celery_app
from backend.services.pipeline_runner import run_pipeline_job

logger = logging.getLogger("celery_pipeline")

@celery_app.task(name="backend.tasks.pipeline.process_pdf_job")
def process_pdf_job(job_id: str):
    """Celery background worker task for running document understanding pipeline."""
    logger.info(f"[Celery] Received task for job: {job_id}")
    try:
        run_pipeline_job(job_id)
        logger.info(f"[Celery] Completed task for job: {job_id}")
    except Exception as e:
        logger.error(f"[Celery] Task for job {job_id} failed: {e}")
        raise e
