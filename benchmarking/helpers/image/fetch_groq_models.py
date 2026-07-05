import os
import sys
import urllib.request
import json
import yaml

def load_env(env_path):
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        os.environ[parts[0].strip()] = parts[1].strip()

def main():
    config_path = "D:/antigravity/benchmarking/vlm_benchmark/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_file = config["paths"]["env_file"]
    load_env(env_file)
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in environment or .env file.")
        sys.exit(1)
        
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )
    
    print("Checking active models on Groq...")
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)
            models = [m["id"] for m in data.get("data", [])]
            print(f"Fetched {len(models)} active models:")
            for m in models:
                print(f" - {m}")
            
            # Check configured models
            configured_groq = config["models"]["groq"]
            print("\nVerification status of configured Groq models:")
            for model_id in configured_groq:
                is_active = model_id in models
                status = "ACTIVE" if is_active else "DEPRECATED/INACTIVE"
                print(f" - {model_id}: {status}")
    except Exception as e:
        print(f"ERROR: Failed to fetch Groq models: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
