import os
import logging
from fastapi import APIRouter, Depends, HTTPException
from backend.core.config import settings, PROJECT_ROOT
from backend.api.schemas import SettingsSchema, SettingsUpdateSchema

logger = logging.getLogger("settings_route")

router = APIRouter()

def mask_key(key: str) -> str:
    """Mask key value into sk-...xxxx standard for front-end rendering security."""
    if not key:
        return ""
    if len(key) <= 8:
        return "********"
    return f"{key[:6]}...{key[-4:]}"

@router.get("/settings", response_model=SettingsSchema)
def get_settings():
    """Retrieve API settings with keys masked for client-side rendering security."""
    return SettingsSchema(
        groq_api_key=mask_key(settings.GROQ_API_KEY),
        gemini_api_key=mask_key(settings.GEMINI_API_KEY),
        openai_api_key=mask_key(settings.OPENAI_API_KEY),
        qianfan_api_key=mask_key(settings.QIANFAN_API_KEY),
        qianfan_secret_key=mask_key(settings.QIANFAN_SECRET_KEY),
        landing_ai_api_key=mask_key(settings.LANDING_AI_API_KEY)
    )

@router.put("/settings", response_model=SettingsSchema)
def update_settings(payload: SettingsUpdateSchema):
    """
    Update server-side .env file with new keys. Masked keys containing '...' 
    are ignored to prevent overwriting active credentials with placeholders.
    """
    env_path = os.path.join(PROJECT_ROOT, ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updates = {
        "GROQ_API_KEY": payload.groq_api_key,
        "GEMINI_API_KEY": payload.gemini_api_key,
        "OPENAI_API_KEY": payload.openai_api_key,
        "QIANFAN_API_KEY": payload.qianfan_api_key,
        "QIANFAN_SECRET_KEY": payload.qianfan_secret_key,
        "LANDING_AI_API_KEY": payload.landing_ai_api_key
    }


    # Filter out None and values that are masked (e.g. contain "...")
    valid_updates = {
        k: v for k, v in updates.items() 
        if v is not None and "..." not in v
    }

    if not valid_updates:
        # No actual changes submitted, return current
        return get_settings()

    updated_keys = set()
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            parts = stripped.split("=", 1)
            k = parts[0].strip()
            if k in valid_updates:
                new_lines.append(f"{k}={valid_updates[k]}\n")
                updated_keys.add(k)
                # Update in-memory settings instance
                setattr(settings, k, valid_updates[k])
                continue
        new_lines.append(line)

    # Append any key that wasn't in the original .env
    for k, v in valid_updates.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}\n")
            setattr(settings, k, v)

    try:
        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        logger.info(f"Successfully updated .env file. Modified keys: {list(valid_updates.keys())}")
    except Exception as e:
        logger.error(f"Failed to write to .env file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save settings to server storage.")

    return get_settings()
