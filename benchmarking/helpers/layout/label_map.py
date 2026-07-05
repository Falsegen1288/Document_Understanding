# layout_benchmark/label_map.py
import json
import os

def check_label_map():
    # 1. Load GT
    gt_path = r"d:\antigravity\benchmarking\MedCore_GT_v2.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
        
    gt_classes = set()
    for page in gt_data["pages"]:
        for elem in page["elements"]:
            gt_classes.add(elem["class"])
            
    print("=" * 55)
    print("  GROUND TRUTH DISTINCT CLASSES")
    print("=" * 55)
    for c in sorted(gt_classes):
        print(f"  - {c}")
    print()

    # 2. Load model predictions cache
    cache_path = r"d:\antigravity\benchmarking\results\evaluation\predictions_cache.json"
    if not os.path.exists(cache_path):
        print(f"Predictions cache not found at: {cache_path}")
        return
        
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
        
    # Get distinct labels from DocLayoutYOLO (docling)
    dl_labels = set()
    for page_preds in cache["docling"]:
        for pred in page_preds:
            dl_labels.add(pred["label"])
            
    # Get distinct labels from Nemotron
    nm_labels = set()
    for page_preds in cache["nemotron"]:
        for pred in page_preds:
            nm_labels.add(pred["label"])
            
    print("=" * 55)
    print("  RAW MODEL PREDICTION LABELS")
    print("=" * 55)
    print("  DocLayoutYOLO (docling) raw labels:")
    for l in sorted(dl_labels):
        print(f"    - {l}")
        
    print("\n  Nemotron-Parse raw labels:")
    for l in sorted(nm_labels):
        print(f"    - {l}")
    print()

    # 3. Define CLASS_MAP
    # Map raw model labels to canonical GT labels
    CLASS_MAP = {
        # Already partially canonicalized or normalized strings:
        'figure': 'figure',
        'list': 'list',
        'paragraph': 'paragraph',
        'section_header': 'section_header',
        'table': 'table',
        'caption': 'caption',
        'footer': 'footer',
        'header': 'header',
        'title': 'title',
        'footnote': 'footnote',
        'formula': 'formula',
        
        # DocLayoutYOLO raw -> GT
        'text': 'paragraph',
        'list_item': 'list',
        'picture': 'figure',
        'page_header': 'header',
        'page_footer': 'footer',
        'logo': 'figure',
        
        # Nemotron raw -> GT
        'Title': 'title',
        'Section-header': 'section_header',
        'Text': 'paragraph',
        'List-item': 'list',
        'Table': 'table',
        'Picture': 'figure',
        'Caption': 'caption',
        'Footnote': 'footnote',
        'Formula': 'formula',
        'Page-header': 'header',
        'Page-footer': 'footer'
    }
    
    print("=" * 55)
    print("  VERIFIED MAPPING TO CANONICAL TAXONOMY")
    print("=" * 55)
    for model_name, labels in [("DocLayoutYOLO", dl_labels), ("Nemotron", nm_labels)]:
        print(f"  {model_name} mappings:")
        for l in sorted(labels):
            mapped = CLASS_MAP.get(l)
            if mapped:
                status = "[OK] mapped to: " + mapped
            else:
                status = "[UNMAPPED]"
            print(f"    {l:18s} -> {status}")
            
    # Save the mapping file for import in benchmark runs
    mapping_data = {
        "CLASS_MAP": CLASS_MAP,
        "GT_CLASSES": list(gt_classes)
    }
    out_map_path = r"d:\antigravity\benchmarking\layout_benchmark\label_map_config.json"
    with open(out_map_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)
    print(f"\n  Saved mapping configuration to {out_map_path}")
    print("=" * 55)

if __name__ == "__main__":
    check_label_map()
