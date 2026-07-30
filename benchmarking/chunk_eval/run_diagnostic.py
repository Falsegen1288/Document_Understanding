"""Deconstruct pipeline output and analyze elements, tables, visual_captions, reading order, duplicates, OCR, and confidence."""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import re
import numpy as np
from pathlib import Path

# Add project root to path for imports if needed
sys.path.append(os.path.abspath('c:/Users/user/Downloads/Document_Understanding'))

outputs_dir = Path("outputs")
docs = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]

def get_bbox_overlap(box1, box2):
    # Intersection over Union
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def run_diagnostics():
    loaded_docs = {}
    for doc_name in docs:
        json_path = outputs_dir / doc_name / f"{doc_name}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_docs[doc_name] = json.load(f)
                
    # --- 1. Inspect metadata ---
    print("=================== 1. METADATA INSPECTION ===================")
    for name, data in loaded_docs.items():
        print(f"\nDocument: {name}")
        print(json.dumps(data.get("metadata", {}), indent=2))
        
    # --- 2. Inspect tables and visual_captions arrays ---
    print("\n=================== 2. TABLES & VISUAL_CAPTIONS ENTRIES ===================")
    target_doc = "Scientific_001"
    doc_data = loaded_docs.get(target_doc)
    if doc_data:
        tables = doc_data.get("tables", [])
        if tables:
            print(f"\nTable entry keys from '{target_doc}':", list(tables[0].keys()))
            print("Full table entry sample:")
            print(json.dumps(tables[0], indent=2))
        else:
            print(f"No tables found in '{target_doc}'")
            
        captions = doc_data.get("visual_captions", [])
        if captions:
            print(f"\nVisual caption entry keys from '{target_doc}':", list(captions[0].keys()))
            print("Full visual caption entry sample:")
            print(json.dumps(captions[0], indent=2))
        else:
            print(f"No visual captions found in '{target_doc}'")

    # --- 3. Cross-reference match rates ---
    print("\n=================== 3. CROSS-REFERENCE MATCH RATES ===================")
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        tables = data.get("tables", [])
        captions = data.get("visual_captions", [])
        
        # Table elements matching
        table_elements = [el for el in elements if el["type"] == "table"]
        matched_tables = 0
        for el in table_elements:
            best_overlap = 0.0
            for t in tables:
                t_bbox = t.get("bbox")
                if t_bbox:
                    overlap = get_bbox_overlap(el["bbox"], t_bbox)
                    best_overlap = max(best_overlap, overlap)
            # Table match rate could also be based on distance of centers or page matching
            # Let's count how many tables match by page and overlap > 0.0
            has_match = False
            for t in tables:
                t_page = t.get("page")
                if t_page == el["page"]:
                    has_match = True
                    break
            if has_match:
                matched_tables += 1
                
        print(f"[{name}] Tables: {matched_tables}/{len(table_elements)} table elements matched to a tables[] entry")

        # Figure elements matching
        figure_elements = [el for el in elements if el["type"] == "figure"]
        matched_figures = 0
        for el in figure_elements:
            has_match = False
            for c in captions:
                c_page = c.get("page")
                if c_page == el["page"]:
                    # check overlap
                    c_bbox = c.get("bbox")
                    if c_bbox and get_bbox_overlap(el["bbox"], c_bbox) > 0.0:
                        has_match = True
                        break
            if has_match:
                matched_figures += 1
        print(f"[{name}] Figures: {matched_figures}/{len(figure_elements)} figure elements matched to a visual_captions[] entry")

    # --- 4. Reading order check ---
    print("\n=================== 4. READING ORDER CHECK ===================")
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        pages = sorted(set(el["page"] for el in elements))
        
        same_order_pages = 0
        total_pages = len(pages)
        diverging_examples = []
        
        for p in pages:
            page_els = [el for el in elements if el["page"] == p]
            # List order
            list_order = [elements.index(el) for el in page_els]
            # y0 sorted order (top coordinate is bbox[1])
            y0_sorted = sorted(page_els, key=lambda el: el["bbox"][1])
            y0_order = [elements.index(el) for el in y0_sorted]
            
            if list_order == y0_order:
                same_order_pages += 1
            else:
                diverging_examples.append((p, page_els, y0_sorted))
                
        pct = (same_order_pages / total_pages) * 100 if total_pages else 0.0
        print(f"[{name}] {same_order_pages}/{total_pages} pages ({pct:.1f}%) have list order == y0-sorted order")
        
        if len(diverging_examples) > 0:
            print(f"  Showing up to 2 diverging pages for '{name}':")
            for p, list_els, y0_els in diverging_examples[:2]:
                print(f"    --- Page {p} ---")
                print(f"    {'List Order (Actual)':<45} | {'y0-Sorted Order (Vertical)':<45}")
                print(f"    {'-'*45} | {'-'*45}")
                for idx in range(max(len(list_els), len(y0_els))):
                    l_text = f"[{list_els[idx]['type']}] {list_els[idx]['content'][:30]}" if idx < len(list_els) else ""
                    y_text = f"[{y0_els[idx]['type']}] {y0_els[idx]['content'][:30]}" if idx < len(y0_els) else ""
                    l_y0 = f" (y0={list_els[idx]['bbox'][1]:.1f})" if idx < len(list_els) else ""
                    y_y0 = f" (y0={y0_els[idx]['bbox'][1]:.1f})" if idx < len(y0_els) else ""
                    print(f"    {l_text + l_y0:<45} | {y_text + y_y0:<45}")

    # --- 5. Duplicate check (page, bbox) ---
    print("\n=================== 5. DUPLICATE CHECK ===================")
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        seen = {}
        duplicates = []
        for idx, el in enumerate(elements):
            key = (el["page"], tuple(el["bbox"]))
            if key in seen:
                duplicates.append((idx, seen[key], key))
            else:
                seen[key] = idx
        print(f"[{name}] Found {len(duplicates)} true duplicates")
        for idx, orig, key in duplicates:
            print(f"  Element {idx} is a duplicate of Element {orig} at Page {key[0]} Bbox {key[1]}")

    # --- 6. OCR corruption breakdown ---
    print("\n=================== 6. OCR CORRUPTION BREAKDOWN ===================")
    flagged_examples = []
    
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        total_chars = 0
        flagged_chars = 0
        
        ligature_count = 0
        hyphen_count = 0
        
        for idx, el in enumerate(elements):
            text = el.get("content", "") or ""
            total_chars += len(text)
            
            is_ligature = False
            is_hyphen = False
            
            # Ligature check: "ﬁ", "ﬂ", "ﬀ" with spaces
            ligatures = re.findall(r'[ﬁﬂﬀ]\s+\w+', text)
            if ligatures:
                is_ligature = True
                ligature_count += 1
                
            # Stray hyphen check: word-hyphen-space-word
            hyphens = re.findall(r'\b\w+-\s+\w+\b', text)
            if hyphens:
                is_hyphen = True
                hyphen_count += 1
                
            if is_ligature or is_hyphen:
                flagged_chars += len(text)
                flagged_examples.append((name, idx, text))
                
        frac = flagged_chars / total_chars if total_chars else 0.0
        print(f"[{name}] Elements Checked: {len(elements)}")
        print(f"  Ligature spacing errors: {ligature_count}")
        print(f"  Stray hyphen-space errors: {hyphen_count}")
        print(f"  Flagged chars fraction: {flagged_chars}/{total_chars} ({frac:.2%})")

    print(f"\nShowing 10 random flagged OCR examples in context:")
    import random
    random.seed(42) # determinism
    if flagged_examples:
        sampled = random.sample(flagged_examples, min(10, len(flagged_examples)))
        for idx, (doc, el_idx, text) in enumerate(sampled, start=1):
            # Clean non-encodable chars before printing just in case
            safe_text = text.replace('\ufffd', '[REPLACEMENT_CHAR]')
            print(f"  Example {idx} | Document: {doc} | Element: {el_idx}")
            print(f"    Content: '{safe_text.strip()}'\n")

    # --- 7. Section/heading signal check ---
    print("=================== 7. HEADING SIGNAL CHECK ===================")
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        title_elements = [el for el in elements if el["type"] == "title"]
        print(f"[{name}] Title-type elements count: {len(title_elements)}")
        if title_elements:
            print("  5 Example title elements with raw attributes:")
            for idx, el in enumerate(title_elements[:5]):
                print(f"    Sample {idx+1}: {el}")

    # --- 8. Confidence field relevance ---
    print("\n=================== 8. CONFIDENCE FIELD RELEVANCE ===================")
    all_confs = []
    flagged_confs = []
    
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        for el in elements:
            conf = el.get("confidence")
            if conf is not None:
                all_confs.append(conf)
                
                text = el.get("content", "") or ""
                is_ligature = bool(re.search(r'[ﬁﬂﬀ]\s+\w+', text))
                is_hyphen = bool(re.search(r'\b\w+-\s+\w+\b', text))
                if is_ligature or is_hyphen:
                    flagged_confs.append(conf)
                    
    if all_confs:
        print(f"All elements confidence distribution: Min={np.min(all_confs):.4f}, Max={np.max(all_confs):.4f}, Mean={np.mean(all_confs):.4f}")
    if flagged_confs:
        print(f"Flagged elements confidence distribution: Min={np.min(flagged_confs):.4f}, Max={np.max(flagged_confs):.4f}, Mean={np.mean(flagged_confs):.4f}")
    else:
        print("No confidence score statistics for flagged elements.")

if __name__ == "__main__":
    run_diagnostics()
