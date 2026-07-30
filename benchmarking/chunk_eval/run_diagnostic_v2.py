"""Run detailed deconstruct on duplicates, precise OCR char fraction, and ligature space checking."""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import re
from pathlib import Path

outputs_dir = Path("outputs")
docs = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]

def run_diagnostic_v2():
    loaded_docs = {}
    for doc_name in docs:
        json_path = outputs_dir / doc_name / f"{doc_name}.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_docs[doc_name] = json.load(f)

    # 1. Print duplicate details
    print("=================== 1. DUPLICATE DETAILS ===================")
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        seen = {}
        duplicates = []
        for idx, el in enumerate(elements):
            key = (el["page"], tuple(el["bbox"]))
            if key in seen:
                duplicates.append((idx, seen[key]))
            else:
                seen[key] = idx
                
        if duplicates:
            print(f"\nDocument: {name} (Found {len(duplicates)} true duplicates)")
            for idx, orig in duplicates:
                el_orig = elements[orig]
                el_dup = elements[idx]
                same_page = el_orig["page"] == el_dup["page"]
                print(f"  Pair:")
                print(f"    Original Element [{orig}]: Page {el_orig['page']} | Bbox: {el_orig['bbox']} | Conf: {el_orig.get('confidence')}")
                print(f"    Duplicate Element [{idx}]: Page {el_dup['page']} | Bbox: {el_dup['bbox']} | Conf: {el_dup.get('confidence')}")
                print(f"    Same page? {same_page}")
        else:
            print(f"\nDocument: {name} - No duplicates found.")

    # 2. Recompute OCR "flagged chars fraction"
    print("\n=================== 2. RECOMPUTED OCR FLAGGED CHARS FRACTION ===================")
    # Regex for stray hyphen-space: word-hyphen-space-word
    # e.g., \b\w+-\s+\w+\b
    hyphen_pattern = re.compile(r'\b\w+-\s+\w+\b')
    
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        total_chars = 0
        hyphen_chars_in_spans = 0
        
        for idx, el in enumerate(elements):
            text = el.get("content", "") or ""
            total_chars += len(text)
            
            # Find all matched spans
            matches = hyphen_pattern.findall(text)
            for m in matches:
                hyphen_chars_in_spans += len(m)
                
        frac = hyphen_chars_in_spans / total_chars if total_chars else 0.0
        print(f"[{name}] Total chars: {total_chars} | Chars in hyphen spans: {hyphen_chars_in_spans} | Fraction: {frac:.4%}")

    # 3. Ligature space search
    print("\n=================== 3. LIGATURE SPACE SEARCH ===================")
    # Ligature characters: ﬁ, ﬂ, ﬀ, ﬃ, ﬄ followed by space
    # Unicode codepoints: \ufb01 (ﬁ), \ufb02 (ﬂ), \ufb00 (ﬀ), \ufb03 (ﬃ), \ufb04 (ﬄ)
    ligature_pattern = re.compile(r'[ﬁﬂﬀﬃﬄ]\s+')
    
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        ligature_matches_count = 0
        matching_snippets = []
        
        for idx, el in enumerate(elements):
            text = el.get("content", "") or ""
            matches = ligature_pattern.findall(text)
            if matches:
                ligature_matches_count += len(matches)
                # Find the text context around matches
                for match_obj in ligature_pattern.finditer(text):
                    start = max(0, match_obj.start() - 20)
                    end = min(len(text), match_obj.end() + 20)
                    matching_snippets.append((idx, text[start:end]))
                    
        print(f"[{name}] Found {ligature_matches_count} ligature-space errors")
        if matching_snippets:
            print("  Examples:")
            for el_idx, snippet in matching_snippets[:5]:
                print(f"    Element [{el_idx}]: ...{snippet.strip()}...")
                
    # Check if there are raw ligatures (without space) and print total occurrences of ligatures
    print("\n=================== Raw Ligature (No Space Required) Prevalence ===================")
    raw_ligature_pattern = re.compile(r'[ﬁﬂﬀﬃﬄ]')
    for name, data in loaded_docs.items():
        elements = data.get("elements", [])
        total_ligatures = 0
        for el in elements:
            text = el.get("content", "") or ""
            matches = raw_ligature_pattern.findall(text)
            total_ligatures += len(matches)
        print(f"[{name}] Total raw ligatures in text: {total_ligatures}")

if __name__ == "__main__":
    run_diagnostic_v2()
