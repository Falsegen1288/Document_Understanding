# layout_benchmark/run_traditional_benchmark.py
import os
import json
import pandas as pd
import sys

# Add parent directory to path so imports work
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layout_benchmark.metrics.coco_metrics import evaluate_coco_metrics

# Config
PRED_CACHE_DIR = r"C:\kaggle\working\cache\predictions"
DATASET_CACHE_DIR = r"C:\kaggle\working\cache\datasets"
OUTPUT_DIR = r"d:\antigravity\benchmarking\layout_benchmark\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_CLASSES_PER_DATASET = {
    'DocLayNet': {'caption', 'footnote', 'formula', 'list_item', 'page_footer',
                  'page_header', 'picture', 'section_header', 'table', 'text', 'title'},
    'PubLayNet': {'text', 'title', 'list_item', 'picture', 'table'},
    'DocBank': {'text', 'caption', 'formula', 'picture', 'page_footer',
                'list_item', 'section_header', 'table', 'title'}
}

# Label configuration
with open(r"d:\antigravity\benchmarking\layout_benchmark\label_map_config.json", "r") as f:
    config_data = json.load(f)
CLASS_MAP = config_data["CLASS_MAP"]

def run_benchmark():
    print("=" * 65)
    print("  RUNNING TRADITIONAL BENCHMARK")
    print("=" * 65)
    
    datasets = ['DocLayNet', 'PubLayNet', 'DocBank']
    models = ['DocLayoutYOLO', 'Nemotron']
    
    records = []
    
    for ds in datasets:
        # Load GT
        gt_path = os.path.join(DATASET_CACHE_DIR, ds, "gt.json")
        if not os.path.exists(gt_path):
            print(f"Dataset GT not found for {ds} at {gt_path}")
            continue
            
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_list = json.load(f)
            
        valid_cls = VALID_CLASSES_PER_DATASET[ds]
        
        # In the original, DocLayNet and PubLayNet have N_SAMPLES = 30, DocBank has N_SAMPLES = 3
        # Let's count elements in gt_list to know how many pages we have
        n_pages = len(gt_list)
        print(f"\nDataset: {ds} ({n_pages} pages)")
        
        for model in models:
            pred_file = f"{model}___{ds}.json" if model == "DocLayoutYOLO" else f"{model}__{ds}.json"
            pred_path = os.path.join(PRED_CACHE_DIR, pred_file)
            
            if not os.path.exists(pred_path):
                # Fallback to key matching
                pred_file_alt = f"DocLayoutYOLO__{ds}.json" if model == "DocLayoutYOLO" else f"Nemotron__{ds}.json"
                pred_path = os.path.join(PRED_CACHE_DIR, pred_file_alt)
                
            if not os.path.exists(pred_path):
                print(f"  Model predictions not found for {model} on {ds} at {pred_path}")
                continue
                
            with open(pred_path, "r", encoding="utf-8") as f:
                pred_data = json.load(f)
                
            # Convert dictionary format from cache back to list of pages
            # Cache matches page index string keys e.g. "0", "1", ...
            pred_list = []
            for idx in range(n_pages):
                k = str(idx)
                if k in pred_data:
                    pred_list.append(pred_data[k])
                else:
                    pred_list.append([])
                    
            # In traditional track, we map predictions to canonical via CLASS_MAP
            mapped_preds = []
            for page in pred_list:
                mapped_page = []
                for p in page:
                    label = p.get("label", p.get("type", "text"))
                    mapped_label = CLASS_MAP.get(label, label)
                    mapped_page.append({
                        "bbox": p["bbox"],
                        "label": mapped_label,
                        "score": p.get("score", 0.9)
                    })
                mapped_preds.append(mapped_page)
                
            # Traditional track GT is already canonicalized during harvest, but apply map to be safe
            mapped_gts = []
            for page in gt_list:
                mapped_page = []
                for g in page:
                    label = g.get("label", g.get("type", "text"))
                    mapped_label = CLASS_MAP.get(label, label)
                    mapped_page.append({
                        "bbox": g["bbox"],
                        "label": mapped_label
                    })
                mapped_gts.append(mapped_page)
                
            metrics = evaluate_coco_metrics(mapped_gts, mapped_preds, valid_cls)
            
            records.append({
                "Dataset": ds,
                "Model": model,
                "mAP50": metrics["mAP50"],
                "mAP5095": metrics["mAP5095"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["F1"],
                "mean_iou": metrics["mean_iou"]
            })
            
            print(f"  {model:15s} -> mAP@50: {metrics['mAP50']:.4f} | mAP@50:95: {metrics['mAP5095']:.4f} | F1: {metrics['F1']:.4f} | mIoU: {metrics['mean_iou']:.4f}")
            
    df = pd.DataFrame(records)
    out_csv = os.path.join(OUTPUT_DIR, "traditional_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nTraditional benchmark results written to: {out_csv}")
    print("=" * 65)

if __name__ == "__main__":
    run_benchmark()
