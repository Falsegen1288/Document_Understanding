import os
from dotenv import load_dotenv

# Add support for reading from .env file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

class Settings:
    PROJECT_NAME: str = "Document Understanding Web Application"
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    QIANFAN_API_KEY: str = os.getenv("QIANFAN_API_KEY", "")
    QIANFAN_SECRET_KEY: str = os.getenv("QIANFAN_SECRET_KEY", "")
    LANDING_AI_API_KEY: str = os.getenv("LANDING_AI_API_KEY", "")

    
    # Storage Directory Config
    STORAGE_DIR: str = os.path.join(PROJECT_ROOT, "storage")
    UPLOADS_DIR: str = os.path.join(STORAGE_DIR, "uploads")
    PAGES_DIR: str = os.path.join(STORAGE_DIR, "pages")
    RESULTS_DIR: str = os.path.join(STORAGE_DIR, "results")
    
    # SQLite Database Config
    DATABASE_URL: str = f"sqlite:///{os.path.join(STORAGE_DIR, 'jobs.db')}"
    
    # Redis Config for Celery (Optional)
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

settings = Settings()

# Ensure directories exist
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.PAGES_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
