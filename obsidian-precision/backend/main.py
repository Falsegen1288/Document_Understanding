import os
import sys
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root, parent, and workspace root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PARENT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PARENT_ROOT, ".."))

for path in [PROJECT_ROOT, PARENT_ROOT, WORKSPACE_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.core.config import settings
from backend.core.database import init_db
from backend.api.routes import upload, jobs, results, stream, settings as settings_route

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("fastapi_main")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Full-stack API wrapper for Document Understanding Pipeline.",
    version="1.0.0"
)

# Enable CORS for frontend clients (Vite localhost:5173, production localhost:3000)
origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifespan/Startup Database Initialization
@app.on_event("startup")
def on_startup():
    logger.info("Initializing SQLite database tables...")
    init_db()
    
    # Run a background thread task to clean up old job directories
    try:
        from datetime import datetime, timedelta
        from backend.core.database import SessionLocal, Job
        import shutil
        
        db = SessionLocal()
        # Find jobs older than 7 days
        expiration_date = datetime.utcnow() - timedelta(days=7)
        old_jobs = db.query(Job).filter(Job.created_at < expiration_date).all()
        
        if old_jobs:
            logger.info(f"Storage Cleanup: Found {len(old_jobs)} jobs older than 7 days. Purging storage...")
            for oj in old_jobs:
                logger.info(f"Purging files for job {oj.id}...")
                
                # Uploaded PDF
                pdf_path = os.path.join(settings.UPLOADS_DIR, f"{oj.id}.pdf")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    
                # Pages directory
                pages_dir = os.path.join(settings.PAGES_DIR, oj.id)
                if os.path.exists(pages_dir):
                    shutil.rmtree(pages_dir)
                    
                # Results directory
                results_dir = os.path.join(settings.RESULTS_DIR, oj.id)
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                    
                db.delete(oj)
            db.commit()
            logger.info("Storage Cleanup completed successfully.")
        db.close()
    except Exception as cleanup_err:
        logger.warning(f"Background storage cleanup encountered an error: {cleanup_err}")

# Mount API Routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"])
app.include_router(results.router, prefix="/api", tags=["Results"])
app.include_router(stream.router, prefix="/api", tags=["SSE Streaming"])
app.include_router(settings_route.router, prefix="/api", tags=["Settings"])

# serve React production bundle dist/index.html at root endpoint
react_index = os.path.abspath(os.path.join(PARENT_ROOT, "obsidian-precision", "dist", "index.html"))
frontend_index = os.path.join(PARENT_ROOT, "frontend", "index.html")

@app.get("/")
def read_root():
    if os.path.exists(react_index):
        with open(react_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    elif os.path.exists(frontend_index):
        with open(frontend_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>Document Understanding Pipeline is running!</h2><p>Please compile the premium client in <code>obsidian-precision</code> to view the full UI.</p>"
    )

# Mount premium assets from obsidian-precision dist directory
react_assets_dir = os.path.abspath(os.path.join(PARENT_ROOT, "obsidian-precision", "dist", "assets"))
os.makedirs(react_assets_dir, exist_ok=True)
app.mount("/assets", StaticFiles(directory=react_assets_dir), name="assets")

# Mount static screens from codes directory
codes_screens_dir = os.path.abspath(os.path.join(PARENT_ROOT, "..", "codes", "public", "screens"))
if os.path.exists(codes_screens_dir):
    app.mount("/screens", StaticFiles(directory=codes_screens_dir), name="screens")

# Optional: Mount generated files path as static fallback in case clients need direct file urls
if os.path.exists(settings.PAGES_DIR):
    app.mount("/storage/pages", StaticFiles(directory=settings.PAGES_DIR), name="pages")

@app.get("/health")
def health_check():
    return {"status": "healthy", "project": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

