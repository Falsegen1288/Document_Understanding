import os
import json
from backend.core.config import settings

def save_uploaded_pdf(file_content: bytes, job_id: str) -> str:
    """Save an uploaded PDF binary file to storage/uploads/{job_id}.pdf."""
    path = os.path.join(settings.UPLOADS_DIR, f"{job_id}.pdf")
    with open(path, "wb") as f:
        f.write(file_content)
    return path

def get_pdf_path(job_id: str) -> str:
    """Get path to the raw PDF for a job."""
    return os.path.join(settings.UPLOADS_DIR, f"{job_id}.pdf")

def get_job_pages_dir(job_id: str) -> str:
    """Get or create the pages directory for a specific job."""
    dir_path = os.path.join(settings.PAGES_DIR, job_id)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_page_image_path(job_id: str, page_num: int) -> str:
    """Get path to the PNG image of a specific page for a job."""
    pages_dir = get_job_pages_dir(job_id)
    return os.path.join(pages_dir, f"page_{page_num}.png")

def get_job_results_dir(job_id: str) -> str:
    """Get or create the results directory for a specific job."""
    dir_path = os.path.join(settings.RESULTS_DIR, job_id)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def get_result_json_path(job_id: str) -> str:
    """Get path to the full result.json for a job."""
    results_dir = get_job_results_dir(job_id)
    return os.path.join(results_dir, "result.json")

def get_page_result_json_path(job_id: str, page_num: int) -> str:
    """Get path to the single page result.json for a job."""
    results_dir = get_job_results_dir(job_id)
    return os.path.join(results_dir, f"page_{page_num}.json")

def save_result_json(job_id: str, result_dict: dict) -> str:
    """Save the complete job results JSON to storage/results/{job_id}/result.json."""
    path = get_result_json_path(job_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    return path

def load_result_json(job_id: str) -> dict:
    """Load the complete job results JSON from storage/results/{job_id}/result.json."""
    path = get_result_json_path(job_id)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_page_result_json(job_id: str, page_num: int, page_result: dict) -> str:
    """Save a single page's JSON to storage/results/{job_id}/page_{page_num}.json."""
    path = get_page_result_json_path(job_id, page_num)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(page_result, f, ensure_ascii=False, indent=2)
    return path

def load_page_result_json(job_id: str, page_num: int) -> dict:
    """Load a single page's JSON from storage/results/{job_id}/page_{page_num}.json."""
    path = get_page_result_json_path(job_id, page_num)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
