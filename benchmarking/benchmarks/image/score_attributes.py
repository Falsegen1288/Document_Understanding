import os
import json
import numpy as np
import pandas as pd
import yaml

def parse_json_safely(text):
    if not text:
        return None
    text = text.strip()
    
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            text = parts[1].strip()
            
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
    except:
        return None

def normalize_str(s):
    if s is None:
        return ""
    return str(s).strip().lower().replace("-", "").replace("_", "").replace(" ", "")

def evaluate_flat_field(gt_val, pred_val):
    norm_gt = normalize_str(gt_val)
    norm_pred = normalize_str(pred_val)
    
    if not norm_gt and not norm_pred:
        return 1.0, 1.0, 1.0  # both empty/null
    if not norm_gt or not norm_pred:
        return 0.0, 0.0, 0.0  # one is empty
    
    # Check match (direct match or substring containment to be slightly robust to exact wording differences)
    if norm_gt == norm_pred or norm_gt in norm_pred or norm_pred in norm_gt:
        return 1.0, 1.0, 1.0
    return 0.0, 0.0, 0.0

def evaluate_set_field(gt_list, pred_list):
    if not gt_list:
        gt_list = []
    if not pred_list:
        pred_list = []
        
    gt_set = {normalize_str(s) for s in gt_list if s}
    pred_set = {normalize_str(s) for s in pred_list if s}
    
    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gt_set:
        return 0.0, 0.0, 0.0
    if not pred_set:
        return 0.0, 0.0, 0.0
        
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluate_key_specs(gt_dict, pred_dict):
    if not gt_dict:
        gt_dict = {}
    if not pred_dict:
        pred_dict = {}
        
    gt_keys = {normalize_str(k) for k in gt_dict.keys()}
    pred_keys = {normalize_str(k) for k in pred_dict.keys()}
    
    if not gt_keys and not pred_keys:
        return 1.0, 1.0, 1.0
    if not gt_keys:
        return 0.0, 0.0, 0.0
    if not pred_keys:
        return 0.0, 0.0, 0.0
        
    # Match keys
    tp = 0
    fp = 0
    fn = 0
    
    # Map normalized keys to original keys/values
    gt_norm_map = {normalize_str(k): (k, v) for k, v in gt_dict.items()}
    pred_norm_map = {normalize_str(k): (k, v) for k, v in pred_dict.items()}
    
    for norm_k in gt_keys:
        if norm_k in pred_keys:
            gt_orig_k, gt_val = gt_norm_map[norm_k]
            pred_orig_k, pred_val = pred_norm_map[norm_k]
            
            # Check value match
            norm_gt_val = normalize_str(gt_val)
            norm_pred_val = normalize_str(pred_val)
            if norm_gt_val == norm_pred_val or norm_gt_val in norm_pred_val or norm_pred_val in norm_gt_val:
                tp += 1
            else:
                fp += 1  # key matches but value doesn't
        else:
            fn += 1  # key missing in prediction
            
    for norm_k in pred_keys:
        if norm_k not in gt_keys:
            fp += 1  # key not present in ground truth
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def main():
    config_path = "D:/antigravity/benchmarking/vlm_benchmark/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    gt_file_path = config["paths"]["gt_file"]
    with open(gt_file_path, "r") as f:
        gt_data = json.load(f)
        
    images = gt_data["images"]
    predictions_dir = config["paths"]["predictions_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    # Ground truths mapping
    gt_map = {img["image_id"]: img for img in images}
    
    target_files = [
        "groq_meta_llama_llama_4_scout_17b_16e_instruct_predictions.jsonl",
        "groq_qwen_qwen3_6_27b_predictions.jsonl",
        "local_moondream_latest_predictions.jsonl",
        "local_qwen2_5vl_3b_predictions.jsonl"
    ]
    pred_files = [f for f in os.listdir(predictions_dir) if f in target_files]
    
    results = []
    
    for pf in pred_files:
        p_path = os.path.join(predictions_dir, pf)
        print(f"\nScoring Attributes: {pf}...")
        
        preds = []
        with open(p_path, "r") as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))
                    
        if not preds:
            print(f" WARNING: No predictions found in {pf}. Skipping.")
            continue
            
        json_valid_count = 0
        img_scores = []
        
        for p in preds:
            img_id = p["image_id"]
            if img_id not in gt_map:
                continue
                
            gt_img = gt_map[img_id]
            raw_text = p.get("attribute_prediction_raw", "")
            pred_attrs = parse_json_safely(raw_text)
            
            is_valid = pred_attrs is not None
            if is_valid:
                json_valid_count += 1
                
            # If invalid JSON, score is 0 across all fields
            if not is_valid:
                img_scores.append({
                    "product_name": (0.0, 0.0, 0.0),
                    "subcategory": (0.0, 0.0, 0.0),
                    "material": (0.0, 0.0, 0.0),
                    "certifications": (0.0, 0.0, 0.0),
                    "intended_setting": (0.0, 0.0, 0.0),
                    "key_specs": (0.0, 0.0, 0.0),
                })
                continue
                
            # Compute per-field precision, recall, F1
            p_name_res = evaluate_flat_field(gt_img.get("product_name"), pred_attrs.get("product_name"))
            subcat_res = evaluate_flat_field(gt_img.get("subcategory"), pred_attrs.get("subcategory"))
            material_res = evaluate_flat_field(gt_img.get("material"), pred_attrs.get("material"))
            
            certs_res = evaluate_set_field(gt_img.get("certifications"), pred_attrs.get("certifications"))
            settings_res = evaluate_set_field(gt_img.get("intended_setting"), pred_attrs.get("intended_setting"))
            
            specs_res = evaluate_key_specs(gt_img.get("key_specs"), pred_attrs.get("key_specs"))
            
            img_scores.append({
                "product_name": p_name_res,
                "subcategory": subcat_res,
                "material": material_res,
                "certifications": certs_res,
                "intended_setting": settings_res,
                "key_specs": specs_res,
            })
            
        if not img_scores:
            print(" WARNING: No matched predictions found. Skipping.")
            continue
            
        # Calculate macro-average per field
        fields = ["product_name", "subcategory", "material", "certifications", "intended_setting", "key_specs"]
        field_averages = {}
        for f in fields:
            p_vals = [s[f][0] for s in img_scores]
            r_vals = [s[f][1] for s in img_scores]
            f1_vals = [s[f][2] for s in img_scores]
            
            field_averages[f] = {
                "precision": np.mean(p_vals),
                "recall": np.mean(r_vals),
                "f1": np.mean(f1_vals)
            }
            
        # Overall macro-average
        overall_p = np.mean([field_averages[f]["precision"] for f in fields])
        overall_r = np.mean([field_averages[f]["recall"] for f in fields])
        overall_f1 = np.mean([field_averages[f]["f1"] for f in fields])
        
        json_validity_rate = json_valid_count / len(preds)
        
        model_name = preds[0]["model"]
        is_local = "local" in pf
        model_type = "local" if is_local else "groq"
        
        results.append({
            "model": model_name,
            "type": model_type,
            "file": pf,
            "JSON_Validity": json_validity_rate,
            "Precision": overall_p,
            "Recall": overall_r,
            "F1": overall_f1,
            "product_name_F1": field_averages["product_name"]["f1"],
            "subcategory_F1": field_averages["subcategory"]["f1"],
            "material_F1": field_averages["material"]["f1"],
            "certifications_F1": field_averages["certifications"]["f1"],
            "intended_setting_F1": field_averages["intended_setting"]["f1"],
            "key_specs_F1": field_averages["key_specs"]["f1"]
        })
        
        print(f"  Results for {model_name}:")
        print(f"    JSON Validity: {json_validity_rate:.2%}")
        print(f"    Precision:     {overall_p:.4f}")
        print(f"    Recall:        {overall_r:.4f}")
        print(f"    F1 Score:      {overall_f1:.4f}")
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "attribute_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved attribute metrics to: {csv_path}")

if __name__ == "__main__":
    main()
