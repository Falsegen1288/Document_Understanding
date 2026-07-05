from datetime import datetime
import uuid
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from backend.core.config import settings

# Setup SQLite Database Engine
# connect_args={"check_same_thread": False} is required for SQLite in multithreaded FastAPI environments
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    doc_type = Column(String, default="auto")     # medical/financial/scientific/legal/commercial/auto
    status = Column(String, default="queued")   # queued/running/done/failed
    layout_algo = Column(String, default="doclayout_yolo")
    ocr_algo = Column(String, default="easyocr")
    table_algo = Column(String, default="tatr")
    figure_algo = Column(String, default="groq")
    total_pages = Column(Integer, nullable=True)
    pages_done = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    result_path = Column(String, nullable=True)      # path to result.json

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "doc_type": self.doc_type,
            "status": self.status,
            "layout_algo": self.layout_algo,
            "ocr_algo": self.ocr_algo,
            "table_algo": self.table_algo,
            "figure_algo": self.figure_algo,
            "total_pages": self.total_pages,
            "pages_done": self.pages_done,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "result_path": self.result_path
        }

# Dependency to get db session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
