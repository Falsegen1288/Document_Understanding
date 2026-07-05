from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class JobCreateResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    id: str
    filename: str
    doc_type: str
    status: str
    layout_algo: str
    ocr_algo: str
    table_algo: str
    figure_algo: str
    total_pages: Optional[int] = None
    pages_done: int
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None

class SettingsSchema(BaseModel):
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    qianfan_api_key: str = ""
    qianfan_secret_key: str = ""
    landing_ai_api_key: str = ""

class SettingsUpdateSchema(BaseModel):
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    qianfan_api_key: Optional[str] = None
    qianfan_secret_key: Optional[str] = None
    landing_ai_api_key: Optional[str] = None


# Pipeline Result Schemas
class ExtractedContentText(BaseModel):
    type: str = "text"
    content: str
    ocr_engine: Optional[str] = None

class ExtractedContentTable(BaseModel):
    type: str = "table"
    markdown: str
    dataframe_csv: Optional[str] = None
    extractor: str
    rows: Optional[int] = None
    cols: Optional[int] = None

class ExtractedContentFigure(BaseModel):
    type: str = "figure"
    caption_ocr: Optional[str] = None
    vlm_description: Optional[str] = None
    vlm_model: Optional[str] = None
    prompt_schema: Optional[str] = None
    crop_path: Optional[str] = None

class DetectionSchema(BaseModel):
    id: str
    label: str
    class_id: int
    confidence: float
    xyxy: List[float]  # bounding box coords [x1, y1, x2, y2]
    extracted: Optional[Dict[str, Any]] = None  # matches one of ExtractedContent models

class PageResultSchema(BaseModel):
    page_number: int
    page_image_path: str
    page_width_px: int
    page_height_px: int
    detections: List[DetectionSchema]

class FullResultSchema(BaseModel):
    job_id: str
    filename: str
    doc_type: str
    pipeline: Dict[str, str]
    pages: List[PageResultSchema]
