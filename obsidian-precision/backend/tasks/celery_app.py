import os
from celery import Celery
from backend.core.config import settings

# Initialize Celery app
celery_app = Celery(
    "document_understanding",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["backend.tasks.pipeline"]
)

# Optional configuration
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1  # Concurrency limit as per specification (Celery workers default to 1 concurrent job)
)
