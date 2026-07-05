"""
Local Ollama VLM Extractor
--------------------------
Queries local Ollama Vision-Language models (e.g., qwen2.5vl:3b or moondream:latest)
running on http://localhost:11434/api/generate.
"""

import io
import base64
import json
import urllib.request
from PIL import Image

def _encode_image(image: Image.Image) -> str:
    """Encode PIL image as base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def describe_figure(
    image: Image.Image,
    prompt: str | None = None,
    model: str = "qwen2.5vl:3b",
    api_key: str | None = None
) -> dict:
    """
    Sends figure image to a local Ollama model.
    """
    if prompt is None:
        prompt = "Describe this product image in one or two sentences, focusing on what the item is, its material/appearance, and any visible identifying features."

    # Validate model name (only allow the two benchmarked local models)
    if model not in ["qwen2.5vl:3b", "moondream:latest"]:
        # default to qwen2.5vl:3b
        model = "qwen2.5vl:3b"

    try:
        base64_image = _encode_image(image)
        url = "http://localhost:11434/api/generate"
        data = json.dumps({
            "model": model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            description = res_json.get("response", "").strip()
            
        return {
            "description": description,
            "engine": "Local Ollama",
            "model": model,
            "prompt_used": prompt
        }
    except Exception as e:
        print(f"[WARNING] Local Ollama VLM analysis failed: {e}. Falling back to baseline description.")
        return {
            "description": f"Mock local description: Figure showing visual content (parsed via {model}).",
            "engine": "Local Ollama",
            "model": model,
            "prompt_used": prompt,
            "fallback": True
        }
