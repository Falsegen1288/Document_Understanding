import os
import sys
import base64
import json
import time
import urllib.request
import subprocess
import yaml

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpu_memory():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        parts = output.strip().split("\n")[0].split(",")
        used = float(parts[0].strip())
        total = float(parts[1].strip())
        return {"used": used, "total": total}
    except Exception:
        return None

def query_ollama(model, prompt, base64_image):
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
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            latency = time.time() - start_time
            res_json = json.loads(res_body)
            return res_json, latency
    except Exception as e:
        print(f"Error querying Ollama model {model}: {e}")
        return None, time.time() - start_time

def sanitize_model_name(model_name):
    return model_name.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_")

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
        
    gt_file_path = config["paths"]["gt_file"]
    with open(gt_file_path, "r") as f:
        gt_data = json.load(f)
        
    images = gt_data["images"]
    image_dir = config["paths"]["image_dir"]
    predictions_dir = config["paths"]["predictions_dir"]
    os.makedirs(predictions_dir, exist_ok=True)
    
    local_models = config["models"]["local"]
    caption_prompt = config["prompts"]["captioning"]
    attr_prompt = config["prompts"]["attribute_extraction"]
    
    for model in local_models:
        model_slug = sanitize_model_name(model)
        pred_file_path = os.path.join(predictions_dir, f"local_{model_slug}_predictions.jsonl")
        print(f"\nRunning Local model: {model}")
        print(f"Saving predictions to: {pred_file_path}")
        
        # Track initial GPU VRAM usage before processing
        gpu_info_before = get_gpu_memory()
        
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
                print(f"  - Requesting caption...")
                caption_res, caption_lat = query_ollama(model, caption_prompt, base64_image)
                
                if caption_res:
                    caption_text = caption_res.get("response", "")
                    caption_eval_count = caption_res.get("eval_count", 0)
                else:
                    caption_text = ""
                    caption_eval_count = 0
                    
                # 2. Query Attributes
                print(f"  - Requesting attributes...")
                attr_res, attr_lat = query_ollama(model, attr_prompt, base64_image)
                
                if attr_res:
                    attr_text = attr_res.get("response", "")
                    attr_eval_count = attr_res.get("eval_count", 0)
                    parsed_attrs = parse_json_safely(attr_text)
                else:
                    attr_text = ""
                    attr_eval_count = 0
                    parsed_attrs = None
                
                gpu_info_during = get_gpu_memory()
                
                prediction_entry = {
                    "image_id": image_id,
                    "image_file": image_file,
                    "caption_prediction": caption_text,
                    "caption_latency": caption_lat,
                    "caption_eval_count": caption_eval_count,
                    "attribute_prediction_raw": attr_text,
                    "attribute_prediction": parsed_attrs,
                    "attribute_latency": attr_lat,
                    "attribute_eval_count": attr_eval_count,
                    "gpu_vram_info": gpu_info_during,
                    "model": model
                }
                
                pred_file.write(json.dumps(prediction_entry) + "\n")
                pred_file.flush()
                print(f"  Completed in {caption_lat + attr_lat:.2f}s")
                
        # Query final GPU memory to log peak usage
        gpu_info_after = get_gpu_memory()
        if gpu_info_before and gpu_info_after:
            vram_delta = gpu_info_after["used"] - gpu_info_before["used"]
            print(f"Model VRAM stats: peak {gpu_info_after['used']}MB, delta {vram_delta}MB")

if __name__ == "__main__":
    main()
