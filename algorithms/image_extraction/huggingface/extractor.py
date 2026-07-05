"""
HuggingFace Inference API Vision Extractor
-------------------------------------------
Uses HuggingFace's free Serverless Inference API with state-of-the-art
open-source vision models for image captioning. No API key required for
public models (rate-limited).
"""

import io
import base64
import os
from PIL import Image

def _encode_image_base64(image: Image.Image) -> str:
    """Encode PIL image as base64 data URI."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def describe_figure(image: Image.Image, prompt: str | None = None, api_key: str | None = None) -> dict:
    """
    Sends figure image to HuggingFace Inference API using a free vision model.
    
    Uses the free Serverless Inference API. An HF token is optional but
    recommended for higher rate limits (set HF_API_KEY in .env).
    """
    key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY") or ""
    
    if prompt is None:
        prompt = "Analyze this figure from a document. Describe what it shows, any data presented, labels, axes, and key takeaways."

    try:
        from huggingface_hub import InferenceClient
        
        client = InferenceClient(token=key if key else None)
        
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image_base64(image)}"}}
                    ]
                }
            ],
            max_tokens=512,
        )
        
        description = response.choices[0].message.content.strip()
        
        return {
            "description": description,
            "engine": "HuggingFace",
            "model": "Llama-3.2-11B-Vision-Instruct",
            "prompt_used": prompt
        }
    except Exception as e:
        print(f"[ERROR] HuggingFace Vision analysis failed: {e}")
        return {
            "description": f"HuggingFace Vision analysis failed: {e}",
            "engine": "HuggingFace",
            "model": "Llama-3.2-11B-Vision-Instruct",
            "error": str(e)
        }
