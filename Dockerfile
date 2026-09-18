FROM python:3.12-slim

# Install system dependencies required for PDF rendering, OpenCV, OCR, and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies (plus fastapi production server requirements)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy celery redis python-multipart gunicorn

# Copy all codebase files into workdir
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Default command starts the API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
