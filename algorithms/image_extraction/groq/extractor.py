"""
Groq Vision API Extractor
--------------------------
Sends diagram/figure crops to Groq vision models to generate captions.
"""

import io
import base64
from PIL import Image
from algorithms.config import GROQ_API_KEY

def _encode_image(image: Image.Image) -> str:
    """Encode PIL image as base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def describe_figure(
    image: Image.Image,
    prompt: str | None = None,
    api_key: str | None = None,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
) -> dict:
    """
    Sends figure image to Groq vision model.
    """
    key = api_key if api_key else GROQ_API_KEY
    if not key:
        return {
            "description": "Groq API key not found. Please add GROQ_API_KEY to your .env file.",
            "engine": "Groq",
            "model": model,
            "error": "ApiKeyMissing"
        }
        
    if prompt is None:
        prompt = "Analyze this figure from a document. Describe what it shows, any data presented, labels, axes, and key takeaways."

    # Validate model
    if model not in ["meta-llama/llama-4-scout-17b-16e-instruct", "qwen/qwen3.6-27b"]:
        model = "meta-llama/llama-4-scout-17b-16e-instruct"

    try:
        from groq import Groq
        client = Groq(api_key=key)
        base64_image = _encode_image(image)
        
        # Call Groq vision model
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model=model
        )
        
        return {
            "description": chat_completion.choices[0].message.content.strip(),
            "engine": "Groq",
            "model": model,
            "prompt_used": prompt
        }
    except Exception as e:
        print(f"[ERROR] Groq Vision analysis failed: {e}")
        return {
            "description": f"Groq Vision analysis failed: {e}",
            "engine": "Groq",
            "model": model,
            "error": str(e)
        }

