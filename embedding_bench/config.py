import os
import yaml
from pathlib import Path

REGISTRY_PATH = Path(__file__).parent / "registry" / "models.yaml"

def load_models_config(path: Path = REGISTRY_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Registry configuration not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "models" not in data:
        raise ValueError("Registry config must contain 'models' key")
    
    # Simple schema validation
    for model_key, model_cfg in data["models"].items():
        required_keys = ["backend_class", "modality", "dim", "max_batch", "cost_per_1m_tokens"]
        for k in required_keys:
            if k not in model_cfg:
                raise ValueError(f"Model '{model_key}' configuration missing required key '{k}'")
                
    return data["models"]
