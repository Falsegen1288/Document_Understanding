import backend.patch_transformers
import os
from dotenv import load_dotenv
load_dotenv()
import json
import uuid
import asyncio
import io
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import shutil

from backend.pipeline import run_pipeline, sessions_progress

app = FastAPI(title="Layout Annotator API")

# Configure CORS so Vite frontend on port 3000/5173 can query the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SaveRequest(BaseModel):
    pages_data: dict # Maps string page numbers to PageData models

class SavePageRequest(BaseModel):
    page: int
    detections: list

@app.post("/api/session/start")
async def start_session(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    pages: str = Form(...) # comma-separated and/or ranges (e.g. "1-5,7,15-20")
):
    # 1. Generate unique session ID
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # 2. Save uploaded files
    pdf_path = os.path.join(session_dir, pdf.filename)
    
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(pdf.file, buffer)
        
    # 4. Parse page numbers
    parsed_pages = parse_page_numbers(pages)
    if not parsed_pages:
        raise HTTPException(status_code=400, detail="Invalid page numbers input format.")
        
    # 5. Launch pipeline background task
    background_tasks.add_task(
        run_pipeline,
        session_id=session_id,
        pdf_path=pdf_path,
        pages=parsed_pages,
        output_dir=session_dir
    )
    
    return {
        "session_id": session_id,
        "models_found": ["DocLayoutYOLO", "Nemotron-Parse-v1.1"],
        "pages": parsed_pages,
        "landing_ai_key_detected": os.environ.get("LANDING_AI_API_KEY") is not None
    }

@app.get("/api/session/{session_id}/status")
async def get_status_stream(session_id: str):
    """
    SSE stream endpoint pushing pipeline progress in real-time to the frontend.
    """
    if session_id not in sessions_progress:
        sessions_progress[session_id] = []
        
    async def event_generator():
        last_idx = 0
        try:
            while True:
                # Yield any new progress logs
                if last_idx < len(sessions_progress[session_id]):
                    for i in range(last_idx, len(sessions_progress[session_id])):
                        data = sessions_progress[session_id][i]
                        yield f"data: {json.dumps(data)}\n\n"
                        
                        if data.get("step") in ["complete", "error"]:
                            await asyncio.sleep(0.5)
                            return
                    last_idx = len(sessions_progress[session_id])
                    
                await asyncio.sleep(0.5)
        except Exception as e:
            import traceback
            err_data = {
                "step": "error",
                "message": f"Status stream error: {str(e)}",
                "traceback": traceback.format_exc(),
                "percent": -1
            }
            yield f"data: {json.dumps(err_data)}\n\n"
        finally:
            return
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/session/{session_id}/data")
async def get_session_data(session_id: str):
    session_file = os.path.join(UPLOAD_DIR, session_id, "annotation_input.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="Session layout data not found or still processing.")
        
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@app.post("/api/session/{session_id}/save")
async def save_session_data(session_id: str, request: SaveRequest):
    session_file = os.path.join(UPLOAD_DIR, session_id, "annotation_input.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="Session does not exist.")
        
    try:
        # We merge/update the ground_truth edits into the stored file on disk
        with open(session_file, "r", encoding="utf-8") as f:
            stored_data = json.load(f)
            
        for page_str, page_val in request.pages_data.items():
            if page_str in stored_data:
                # Update ground_truth field
                stored_data[page_str]["ground_truth"] = page_val.get("ground_truth")
                
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(stored_data, f, indent=2)
            
        return {"status": "success", "message": "Ground truth saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save ground truth: {e}")


# ─── Issue 4: Save single page endpoint ───────────────────────
@app.post("/api/session/{session_id}/save-page")
async def save_page(session_id: str, request: SavePageRequest):
    session_file = os.path.join(UPLOAD_DIR, session_id, "annotation_input.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="Session does not exist.")
    
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            stored_data = json.load(f)
        
        page_str = str(request.page)
        if page_str not in stored_data:
            raise HTTPException(status_code=404, detail=f"Page {request.page} not found in session.")
        
        stored_data[page_str]["ground_truth"] = request.detections
        
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(stored_data, f, indent=2)
        
        return {"status": "saved", "page": request.page}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save page: {e}")


# ─── Issue 5: Commit session → annotated PDF download ─────────
@app.post("/api/session/{session_id}/commit")
async def commit_session(session_id: str):
    from PIL import Image, ImageDraw, ImageFont
    
    session_file = os.path.join(UPLOAD_DIR, session_id, "annotation_input.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="Session does not exist.")
    
    with open(session_file, "r", encoding="utf-8") as f:
        stored_data = json.load(f)
    
    # Color map keyed by DocLayNet type names (lowercase with underscores)
    CLASS_COLORS = {
        'title':          '#E63946',
        'section_header': '#FF6B35',
        'text':           '#457B9D',
        'list_item':      '#2A9D8F',
        'table':          '#E9C46A',
        'picture':        '#F4A261',
        'caption':        '#8ECAE6',
        'footnote':       '#A8DADC',
        'formula':        '#6D6875',
        'page_header':    '#B5838D',
        'page_footer':    '#E5989B',
    }
    
    CLASS_LABELS = {
        'title':          'Title',
        'section_header': 'Section-header',
        'text':           'Text',
        'list_item':      'List-item',
        'table':          'Table',
        'picture':        'Picture',
        'caption':        'Caption',
        'footnote':       'Footnote',
        'formula':        'Formula',
        'page_header':    'Page-header',
        'page_footer':    'Page-footer',
    }
    
    def hex_to_rgba(hex_color: str, alpha: int = 255):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        return (r, g, b, alpha)
    
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    annotated_images = []
    
    # Sort pages numerically
    sorted_pages = sorted(stored_data.keys(), key=lambda x: int(x))
    
    for page_str in sorted_pages:
        page_data = stored_data[page_str]
        page_num = int(page_str)
        
        # Load the original page PNG
        png_path = os.path.join(session_dir, f"page_{page_num:02d}_original.png")
        if not os.path.exists(png_path):
            continue
        
        img = Image.open(png_path).convert("RGBA")
        
        # Get detections: prefer ground_truth, then model_a (DocLayoutYOLO)
        detections = page_data.get("ground_truth")
        if detections is None:
            detections = page_data.get("model_a", {}).get("detections", [])
        
        # Create a transparent overlay for filled rectangles
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw on the base image for borders and text
        draw = ImageDraw.Draw(img)
        
        # Try to load a small font
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except (IOError, OSError):
                font = ImageFont.load_default()
        
        for det in detections:
            det_type = det.get("type", "text")
            bbox = det.get("bbox", [0, 0, 0, 0])
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            
            color_hex = CLASS_COLORS.get(det_type, '#46f1c5')
            fill_rgba = hex_to_rgba(color_hex, alpha=64)   # 25% opacity
            border_rgb = hex_to_rgba(color_hex, alpha=255)[:3]
            
            # Draw filled rectangle on overlay
            overlay_draw.rectangle([x0, y0, x1, y1], fill=fill_rgba)
            
            # Draw border
            for offset in range(2):  # 2px border
                draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=border_rgb)
            
            # Draw label
            label = CLASS_LABELS.get(det_type, det_type.upper())
            try:
                text_bbox = font.getbbox(label)
                tw = text_bbox[2] - text_bbox[0]
                th = text_bbox[3] - text_bbox[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)
            
            pill_h = th + 6
            pill_w = tw + 10
            # Dark background pill
            draw.rectangle([x0, y0 - pill_h, x0 + pill_w, y0], fill=(30, 30, 30, 220))
            draw.text((x0 + 5, y0 - pill_h + 3), label, fill=border_rgb, font=font)
        
        # Composite overlay onto image
        img = Image.alpha_composite(img, overlay)
        # Convert back to RGB for PDF
        annotated_images.append(img.convert("RGB"))
    
    if not annotated_images:
        raise HTTPException(status_code=500, detail="No pages found to render.")
    
    # Compile into PDF
    pdf_buffer = io.BytesIO()
    if len(annotated_images) == 1:
        annotated_images[0].save(pdf_buffer, format="PDF")
    else:
        annotated_images[0].save(
            pdf_buffer,
            format="PDF",
            save_all=True,
            append_images=annotated_images[1:]
        )
    pdf_buffer.seek(0)
    
    short_id = session_id[:8]
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="ground_truth_{short_id}.pdf"'
        }
    )


@app.get("/api/images/{session_id}/{filename}")
async def serve_image(session_id: str, filename: str):
    image_path = os.path.join(UPLOAD_DIR, session_id, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path)

@app.get("/api/session/{session_id}/export")
async def export_ground_truth(session_id: str):
    session_file = os.path.join(UPLOAD_DIR, session_id, "annotation_input.json")
    if not os.path.exists(session_file):
        raise HTTPException(status_code=404, detail="Session does not exist.")
        
    with open(session_file, "r", encoding="utf-8") as f:
        stored_data = json.load(f)
        
    output = {}
    for page_str, page_val in stored_data.items():
        output[page_str] = page_val.get("ground_truth")
        
    return output

@app.delete("/api/session/{session_id}")
async def reset_session(session_id: str):
    """
    Top-bar 'New Session' triggers session clearing: deletes the session temp folder
    and removes progress tracking.
    """
    session_dir = os.path.join(UPLOAD_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    if session_id in sessions_progress:
        del sessions_progress[session_id]
    return {"status": "success", "message": f"Session {session_id} deleted."}

def parse_page_numbers(pages_input: str) -> List[int]:
    """
    Parse a comma-separated list of page numbers and/or ranges.
    Example: '1-3, 5, 7-10' -> [1, 2, 3, 5, 7, 8, 9, 10]
    """
    pages = set()
    cleaned = pages_input.replace(" ", "")
    parts = cleaned.split(",")
    
    for part in parts:
        if not part:
            continue
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                if start <= end:
                    pages.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                pages.add(int(part))
            except ValueError:
                pass
                
    return sorted(list(pages))
