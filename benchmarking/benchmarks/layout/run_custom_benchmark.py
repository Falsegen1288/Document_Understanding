# layout_benchmark/run_custom_benchmark.py
import os
import json
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

# Add parent directory to path so imports work
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layout_benchmark.metrics.layer1_geometric import evaluate_layer1
from layout_benchmark.metrics.layer2_cote import evaluate_layer2
from layout_benchmark.metrics.layer3_led import evaluate_layer3
from layout_benchmark.metrics.layer4_reading_order import evaluate_layer4

# Config
GT_RAW_PATH = r"d:\antigravity\benchmarking\MedCore_GT_v2.json"
GT_TIGHT_PATH = r"d:\antigravity\benchmarking\layout_GT_custom_tightened.json"
PRED_CACHE_PATH = r"d:\antigravity\benchmarking\results\evaluation\predictions_cache.json"
OUTPUT_DIR = r"d:\antigravity\benchmarking\layout_benchmark\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CLASS_MAP
with open(r"d:\antigravity\benchmarking\layout_benchmark\label_map_config.json", "r") as f:
    config_data = json.load(f)
CLASS_MAP = config_data["CLASS_MAP"]

# Document dimensions in points
PAGE_W = 595.28
PAGE_H = 841.89

def load_gt_elements(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pages_gt = {}
    for page_entry in data["pages"]:
        page_num = page_entry["page"]
        elements = []
        for elem in page_entry["elements"]:
            if elem["class"] == "product_card":
                continue
            bbox = elem["bbox_pt"]
            # Convert bottom-left origin to top-left origin
            y0_topleft = PAGE_H - (bbox["y"] + bbox["h"])
            y1_topleft = PAGE_H - bbox["y"]
            
            elements.append({
                "id": elem["id"],
                "bbox": [bbox["x"], min(y0_topleft, y1_topleft), bbox["x"] + bbox["w"], max(y0_topleft, y1_topleft)],
                "label": elem["class"]
            })
        pages_gt[page_num] = elements
    return pages_gt

def run_custom_eval():
    print("=" * 65)
    print("  RUNNING CUSTOM LAYOUT BENCHMARK")
    print("=" * 65)
    
    # 1. Load GT variants
    gt_raw = load_gt_elements(GT_RAW_PATH)
    gt_tight = load_gt_elements(GT_TIGHT_PATH)
    
    # 2. Load prediction cache
    with open(PRED_CACHE_PATH, "r", encoding="utf-8") as f:
        preds_cache = json.load(f)
        
    models = {
        "DocLayoutYOLO": preds_cache["docling"],
        "Nemotron": preds_cache["nemotron"]
    }
    
    gt_variants = {
        "GT-raw": gt_raw,
        "GT-tight": gt_tight
    }
    
    layer1_records = []
    layer2_records = []
    layer3_records = []
    layer4_records = []
    
    # Run combinations
    for gt_name, gt_dict in gt_variants.items():
        print(f"\nEvaluating against: {gt_name}")
        for model_name, pages_preds in models.items():
            print(f"  Model: {model_name}")
            
            # Temporary stores for averaging
            l1_results = []
            l2_results = []
            l3_results = defaultdict(list)
            l4_results = []
            
            # Loop page-by-page (4 pages)
            for page_num in sorted(gt_dict.keys()):
                gt_boxes = gt_dict[page_num]
                # Predictions cache is 0-indexed list for pages 1-4
                pred_boxes = pages_preds[page_num - 1]
                
                # Format predictions correctly
                formatted_preds = []
                for p in pred_boxes:
                    formatted_preds.append({
                        "bbox": p["bbox"],
                        "label": p.get("label", p.get("type", "text")),
                        "score": p.get("score", 0.9)
                    })
                
                # --- Layer 1 ---
                l1 = evaluate_layer1([gt_boxes], [formatted_preds], CLASS_MAP)
                # Compute tolerant version too (Tier 3)
                l1_tol = evaluate_layer1([gt_boxes], [formatted_preds], CLASS_MAP, tolerant=True)
                l1_results.append((l1, l1_tol))
                
                # --- Layer 2 ---
                l2 = evaluate_layer2(gt_boxes, formatted_preds, PAGE_W, PAGE_H)
                l2_results.append(l2)
                
                # --- Layer 3 ---
                l3 = evaluate_layer3(gt_boxes, formatted_preds, CLASS_MAP)
                for k, v in l3.items():
                    l3_results[k].append(v)
                    
                # --- Layer 4 ---
                l4 = evaluate_layer4(gt_boxes, formatted_preds, CLASS_MAP)
                l4_results.append(l4)
                
            # Average across pages
            # Layer 1 averages
            avg_p = np.mean([r[0]["precision"] for r in l1_results])
            avg_r = np.mean([r[0]["recall"] for r in l1_results])
            avg_f1 = np.mean([r[0]["F1"] for r in l1_results])
            avg_iou = np.mean([r[0]["mean_iou"] for r in l1_results])
            avg_acc = np.mean([r[0]["class_acc"] for r in l1_results])
            
            avg_p_tol = np.mean([r[1]["precision"] for r in l1_results])
            avg_r_tol = np.mean([r[1]["recall"] for r in l1_results])
            avg_f1_tol = np.mean([r[1]["F1"] for r in l1_results])
            avg_iou_tol = np.mean([r[1]["mean_iou"] for r in l1_results])
            
            layer1_records.append({
                "GT_Variant": gt_name, "Model": model_name,
                "Precision": avg_p, "Recall": avg_r, "F1": avg_f1, "mean_iou": avg_iou, "Class_Acc": avg_acc,
                "Precision_tol": avg_p_tol, "Recall_tol": avg_r_tol, "F1_tol": avg_f1_tol, "mean_iou_tol": avg_iou_tol
            })
            
            # Layer 2 averages
            avg_cov = np.mean([r["Coverage"] for r in l2_results])
            avg_ov = np.mean([r["Overlap"] for r in l2_results])
            avg_tr = np.mean([r["Trespass"] for r in l2_results])
            avg_ex = np.mean([r["Excess"] for r in l2_results])
            
            layer2_records.append({
                "GT_Variant": gt_name, "Model": model_name,
                "Coverage": avg_cov, "Overlap": avg_ov, "Trespass": avg_tr, "Excess": avg_ex
            })
            
            # Layer 3 averages
            l3_avg = {k: int(round(np.mean(v))) for k, v in l3_results.items()}
            l3_avg["GT_Variant"] = gt_name
            l3_avg["Model"] = model_name
            layer3_records.append(l3_avg)
            
            # Layer 4 averages (ignore NaN values in Kendall's tau)
            tau_vals = [r["tau"] for r in l4_results if not np.isnan(r["tau"])]
            roa_vals = [r["roa"] for r in l4_results if not np.isnan(r["roa"])]
            avg_tau = np.mean(tau_vals) if tau_vals else float('nan')
            avg_roa = np.mean(roa_vals) if roa_vals else float('nan')
            
            layer4_records.append({
                "GT_Variant": gt_name, "Model": model_name,
                "ROKT": avg_tau, "ROA": avg_roa
            })
            
            print(f"    L1: F1={avg_f1:.3f} | ClassAcc={avg_acc:.3f}")
            print(f"    L2: Coverage={avg_cov:.3f} | Excess={avg_ex:.3f}")
            
    # Save CSVs
    pd.DataFrame(layer1_records).to_csv(os.path.join(OUTPUT_DIR, "custom_results_layer1.csv"), index=False)
    pd.DataFrame(layer2_records).to_csv(os.path.join(OUTPUT_DIR, "custom_results_layer2.csv"), index=False)
    pd.DataFrame(layer3_records).to_csv(os.path.join(OUTPUT_DIR, "custom_results_layer3.csv"), index=False)
    pd.DataFrame(layer4_records).to_csv(os.path.join(OUTPUT_DIR, "custom_results_layer4.csv"), index=False)
    
    print("\nCustom layout benchmark execution complete. Layer CSV files generated.")
    print("=" * 65)

if __name__ == "__main__":
    run_custom_eval()
