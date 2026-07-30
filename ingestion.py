"""Ingestion stage for document pipeline. Generates metadata manifest and checks for duplicates."""
import os
import re
import hashlib
import datetime
import json
from pathlib import Path
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "access_config.yaml")

def get_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_access_tags(file_path: str) -> dict:
    """Read access tags from access_config.yaml based on file path."""
    default_tags = {
        "confidentiality": "internal",
        "department": None,
        "allowed_roles": ["all"]
    }
    
    if not os.path.exists(CONFIG_PATH):
        return default_tags
        
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"[WARNING] Failed to load access config: {e}. Using defaults.")
        return default_tags
        
    normalized_path = file_path.replace("\\", "/").lower()
    folder_rules = config.get("folder_rules", {})
    
    for rule_dir, tags in folder_rules.items():
        normalized_rule = rule_dir.replace("\\", "/").lower().strip("/")
        # Check if the folder rule matches any part of the path directories
        if normalized_rule in normalized_path:
            return tags
            
    return config.get("default", default_tags)

def generate_manifest(file_path: str) -> dict:
    """Generate the A1-Lite Ingestion Manifest for a given file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")
        
    content_hash = get_file_hash(file_path)
    file_size = os.path.getsize(file_path)
    
    # Timestamps
    ingested_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    mtime = os.path.getmtime(file_path)
    file_modified_at = datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc).isoformat()
    
    # Sanitization
    filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    sanitized = re.sub(r'[^a-z0-9_]', '_', filename_without_ext.lower())
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    truncated = sanitized[:50]
    
    # Unique collision-safe doc_id
    doc_id = f"{truncated}_{content_hash[:8]}"
    
    # Access control tags
    access_tags = get_access_tags(file_path)
    
    manifest = {
        "doc_id": doc_id,
        "source_filename": os.path.basename(file_path),
        "source_path": os.path.abspath(file_path),
        "content_hash": content_hash,
        "file_size_bytes": file_size,
        "ingested_at": ingested_at,
        "file_modified_at": file_modified_at,
        "access_tags": access_tags,
        "schema_version": "1.0"
    }
    return manifest

def find_existing_manifest(content_hash: str, outputs_root: str) -> tuple:
    """Scan outputs directory to find if a manifest with the same content_hash already exists."""
    outputs_path = Path(outputs_root)
    if not outputs_path.exists():
        return None, None
        
    for manifest_file in outputs_path.glob("**/manifest.json"):
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("content_hash") == content_hash:
                return manifest.get("doc_id"), manifest_file
        except Exception:
            continue
    return None, None
