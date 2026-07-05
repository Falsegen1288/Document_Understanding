import os
import sys
import base64
import json
import time
import urllib.request
import urllib.error
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_groq(model, messages, api_key):
    url = "https://api.groq.com/openai/v1/chat/completions"
    data = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False
    }).encode("utf-8")
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
    )
    
    retries = 6
    backoff = 12  # Start with 12 seconds sleep on 429
    
    for attempt in range(retries):
        start_time = time.time()
        try:
            with urllib.request.urlopen(req) as response:
                res_body = response.read().decode("utf-8")
                latency = time.time() - start_time
                res_json = json.loads(res_body)
                return res_json, latency
        except urllib.error.HTTPError as e:
            latency = time.time() - start_time
            if e.code == 429:
                retry_after_header = e.headers.get("retry-after") or e.headers.get("x-ratelimit-reset")
                sleep_time = backoff
                if retry_after_header:
                    try:
                        if retry_after_header.isdigit():
                            sleep_time = int(retry_after_header)
                        elif retry_after_header.endswith("s"):
                            sleep_time = int(retry_after_header[:-1])
                        elif "m" in retry_after_header:
                            parts = retry_after_header.split("m")
                            minutes = int(parts[0])
                            seconds = int(parts[1].replace("s", "")) if parts[1] else 0
                            sleep_time = minutes * 60 + seconds
                    except:
                        pass
                print(f"  [429 Rate Limit] Retrying in {sleep_time} seconds (attempt {attempt+1}/{retries})...")
                time.sleep(sleep_time)
                backoff *= 2
            else:
                print(f"Error querying Groq model {model}: {e}")
                return None, latency
        except Exception as e:
            print(f"Error querying Groq model {model}: {e}")
            return None, time.time() - start_time
            
    return None, time.time() - start_time

def sanitize_model_name(model_name):
    return model_name.replace("/", "_").replace(".", "_").replace("-", "_")

def parse_json_safely(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx+1]
        
    try:
        return json.loads(text)
    except Exception as e:
        print(f"JSON parse error: {e}")
        return None

def main():
    config_path = "D:/antigravity/benchmarking/vlm_benchmark/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    load_env(config["paths"]["env_file"])
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found.")
        sys.exit(1)
        
    gt_file_path = config["paths"]["gt_file"]
    with open(gt_file_path, "r") as f:
        gt_data = json.load(f)
        
    images = gt_data["images"]
    image_dir = config["paths"]["image_dir"]
    predictions_dir = config["paths"]["predictions_dir"]
    os.makedirs(predictions_dir, exist_ok=True)
    
    groq_models = config["models"]["groq"]
    caption_prompt = config["prompts"]["captioning"]
    attr_prompt = config["prompts"]["attribute_extraction"]
    
    for model in groq_models:
        model_slug = sanitize_model_name(model)
        pred_file_path = os.path.join(predictions_dir, f"groq_{model_slug}_predictions.jsonl")
        print(f"\nRunning Groq model: {model}")
        print(f"Saving predictions to: {pred_file_path}")
        
        with open(pred_file_path, "w") as pred_file:
            for idx, img in enumerate(images):
                image_id = img["image_id"]
                image_file = img["image_file"]
                image_path = os.path.join(image_dir, image_file)
                print(f" [{idx+1}/{len(images)}] Processing {image_id}...")
                
                if not os.path.exists(image_path):
                    print(f"  WARNING: Image not found at {image_path}. Skipping.")
                    continue
                    
                base64_image = encode_image(image_path)
                
                # 1. Query Caption
                caption_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": caption_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ]
                
                print(f"  - Requesting caption...")
                caption_res, caption_lat = query_groq(model, caption_messages, api_key)
                
                if caption_res:
                    caption_text = caption_res["choices"][0]["message"]["content"]
                    caption_tokens = caption_res.get("usage", {})
                else:
                    caption_text = ""
                    caption_tokens = {}
                    
                # Rate limit safety sleep
                time.sleep(2)
                
                # 2. Query Attributes
                attr_messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": attr_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ]
                
                print(f"  - Requesting attributes...")
                attr_res, attr_lat = query_groq(model, attr_messages, api_key)
                
                if attr_res:
                    attr_text = attr_res["choices"][0]["message"]["content"]
                    attr_tokens = attr_res.get("usage", {})
                    parsed_attrs = parse_json_safely(attr_text)
                else:
                    attr_text = ""
                    attr_tokens = {}
                    parsed_attrs = None
                    
                prediction_entry = {
                    "image_id": image_id,
                    "image_file": image_file,
                    "caption_prediction": caption_text,
                    "caption_latency": caption_lat,
                    "caption_tokens": caption_tokens,
                    "attribute_prediction_raw": attr_text,
                    "attribute_prediction": parsed_attrs,
                    "attribute_latency": attr_lat,
                    "attribute_tokens": attr_tokens,
                    "model": model
                }
                
                pred_file.write(json.dumps(prediction_entry) + "\n")
                pred_file.flush()
                print(f"  Completed in {caption_lat + attr_lat:.2f}s")
                
                # Rate limit safety sleep
                time.sleep(4)

if __name__ == "__main__":
    main()
