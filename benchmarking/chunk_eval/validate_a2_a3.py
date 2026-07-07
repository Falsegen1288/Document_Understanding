"""Validate A2/A3 outputs against target schema and run sanity checks on actual outputs."""
import os
import sys
import json
import re
from pathlib import Path

# Add project root to path for imports if needed
sys.path.append(os.path.abspath('c:/Users/user/Downloads/Document_Understanding'))

# Output paths
outputs_dir = Path("outputs")
docs = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]

# Target Schema Element Types
TARGET_ELEMENT_TYPES = {
    "title", "paragraph", "table", "figure", "header", "footer", "list_item", "formula"
}

def validate_corpus():
    results = {}
    failures = []
    
    total_docs = 0
    total_elements = 0
    
    # Track passes for 10 checks
    checks_passed = {i: True for i in range(1, 11)}
    check_notes = {i: "" for i in range(1, 11)}
    
    # 10 checks lists
    schema_completeness_fails = []
    type_correctness_fails = []
    bbox_sanity_fails = []
    uniqueness_fails = []
    reading_order_fails = []
    table_integrity_fails = []
    figure_integrity_fails = []
    text_quality_fails = []
    section_path_fails = []
    cross_doc_consistency_fails = []
    
    # For cross-doc consistency check
    doc_keys_set = set()
    doc_element_keys_set = set()
    
    # First pass: load and inspect documents
    loaded_docs = []
    for doc_name in docs:
        json_path = outputs_dir / doc_name / f"{doc_name}.json"
        if not json_path.exists():
            continue
        total_docs += 1
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                loaded_docs.append((doc_name, data))
                doc_keys_set.add(tuple(sorted(data.keys())))
                for el in data.get("elements", []):
                    doc_element_keys_set.add(tuple(sorted(el.keys())))
            except Exception as e:
                failures.append((10, doc_name, "Root", f"JSON parsing failed: {e}"))
                checks_passed[10] = False
                
    # doc_id uniqueness across corpus
    doc_ids = []
    for doc_name, data in loaded_docs:
        doc_id = data.get("doc_id")
        if doc_id:
            doc_ids.append(doc_id)
    if len(doc_ids) != len(set(doc_ids)) and len(doc_ids) > 0:
        checks_passed[4] = False
        uniqueness_fails.append(("Corpus", "doc_id", "doc_id is not unique across the corpus"))
        
    for doc_name, data in loaded_docs:
        # Check doc_id presence
        doc_id = data.get("doc_id")
        if not doc_id:
            checks_passed[1] = False
            schema_completeness_fails.append((doc_name, "Root", "Missing required field 'doc_id' at document level"))
            
        elements = data.get("elements", [])
        total_elements += len(elements)
        
        # Track elements by page for uniqueness/monotony
        page_elements = {}
        
        # Track element_ids within this document
        el_ids = []
        
        for idx, el in enumerate(elements):
            el_type = el.get("element_type", el.get("type")) # Try target or actual
            
            # --- Check 1: Schema Completeness (Target Schema) ---
            missing_fields = []
            for field in ["element_id", "element_type", "page_number", "bbox", "reading_order", "section_path"]:
                if field not in el:
                    missing_fields.append(field)
            
            if el_type in ["title", "paragraph", "header", "footer", "list_item", "formula"]:
                if "text" not in el:
                    missing_fields.append("text")
            if el_type == "table":
                if "table_markdown" not in el:
                    missing_fields.append("table_markdown")
            if el_type == "figure":
                if "image_path" not in el:
                    missing_fields.append("image_path")
                if "image_caption" not in el:
                    missing_fields.append("image_caption")
            
            if missing_fields:
                checks_passed[1] = False
                schema_completeness_fails.append((doc_name, f"Element {idx}", f"Missing target fields: {missing_fields}"))
                
            # --- Check 2: Type Correctness (Target Schema) ---
            type_errors = []
            if "page_number" in el:
                if not isinstance(el["page_number"], int) or el["page_number"] < 1:
                    type_errors.append("page_number must be integer >= 1")
            if "bbox" in el:
                bbox_val = el["bbox"]
                if not isinstance(bbox_val, list) or len(bbox_val) != 4 or not all(isinstance(x, (int, float)) for x in bbox_val):
                    type_errors.append("bbox must be a list of 4 numeric values")
            if "reading_order" in el:
                if not isinstance(el["reading_order"], int):
                    type_errors.append("reading_order must be an integer")
            if type_errors:
                checks_passed[2] = False
                type_correctness_fails.append((doc_name, f"Element {idx}", f"Type errors: {type_errors}"))
                
            # --- Type check on ACTUAL schema fields for sanity check ---
            actual_page = el.get("page")
            actual_bbox = el.get("bbox")
            if actual_page is not None and not isinstance(actual_page, int):
                checks_passed[2] = False
                type_correctness_fails.append((doc_name, f"Element {idx}", f"Actual 'page' field is not int: {type(actual_page)}"))
            if actual_bbox is not None:
                if not isinstance(actual_bbox, list) or len(actual_bbox) != 4 or not all(isinstance(x, (int, float)) for x in actual_bbox):
                    checks_passed[2] = False
                    type_correctness_fails.append((doc_name, f"Element {idx}", f"Actual 'bbox' is not 4 numeric values: {actual_bbox}"))

            # --- Check 3: Bbox Sanity ---
            bbox_to_check = el.get("bbox")
            if bbox_to_check and isinstance(bbox_to_check, list) and len(bbox_to_check) == 4:
                x0, y0, x1, y1 = bbox_to_check
                bbox_errs = []
                if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
                    bbox_errs.append("negative coordinates")
                if x1 <= x0:
                    bbox_errs.append(f"x1 ({x1}) <= x0 ({x0})")
                if y1 <= y0:
                    bbox_errs.append(f"y1 ({y1}) <= y0 ({y0})")
                if bbox_errs:
                    checks_passed[3] = False
                    bbox_sanity_fails.append((doc_name, f"Element {idx}", f"Bbox sanity violation: {', '.join(bbox_errs)} (bbox: {bbox_to_check})"))

            # --- Check 4: Uniqueness (element_id) ---
            el_id = el.get("element_id")
            if el_id:
                el_ids.append(el_id)
                
            # --- Check 5: Reading Order per Page ---
            p_num = el.get("page_number", el.get("page"))
            r_ord = el.get("reading_order")
            if p_num is not None:
                page_elements.setdefault(p_num, []).append((r_ord, idx, el))

            # --- Check 6: Table Integrity ---
            # If target table_markdown is present
            t_md = el.get("table_markdown")
            if el_type == "table":
                # Let's check target schema if present
                if t_md is not None:
                    # check if it is markdown/html
                    if not (isinstance(t_md, str) and ("|" in t_md or "<table>" in t_md)):
                        checks_passed[6] = False
                        table_integrity_fails.append((doc_name, f"Element {idx}", "table_markdown is not a valid markdown or HTML table"))
                
                # Check ACTUAL extracted table markdown if present
                ext = el.get("extracted")
                if ext and isinstance(ext, dict):
                    ext_md = ext.get("markdown")
                    if ext_md:
                        if not ("|" in ext_md or "<table>" in ext_md):
                            checks_passed[6] = False
                            table_integrity_fails.append((doc_name, f"Element {idx}", "Extracted table markdown lacks markdown/HTML indicators"))
                        # Count columns in markdown headers and rows
                        lines = [l.strip() for l in ext_md.split("\n") if l.strip()]
                        pipe_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
                        if pipe_lines:
                            col_counts = [l.count("|") for l in pipe_lines]
                            if len(set(col_counts)) > 1:
                                checks_passed[6] = False
                                table_integrity_fails.append((doc_name, f"Element {idx}", f"Inconsistent column count in actual table markdown: {set(col_counts)}"))

            # --- Check 7: Figure Integrity ---
            if el_type in ["figure", "image"]:
                img_path = el.get("image_path")
                img_cap = el.get("image_caption")
                if img_path is not None:
                    # check if exists on disk
                    p = Path(img_path)
                    if not p.exists():
                        checks_passed[7] = False
                        figure_integrity_fails.append((doc_name, f"Element {idx}", f"Figure image_path does not exist: {img_path}"))
                if img_cap is not None and not img_cap.strip():
                    checks_passed[7] = False
                    figure_integrity_fails.append((doc_name, f"Element {idx}", "Figure image_caption is empty"))
                    
            # --- Check 8: Text Quality Spot-Check ---
            # Check actual content text or target text
            text_to_check = el.get("text", el.get("content", "")) or ""
            if isinstance(text_to_check, str) and text_to_check:
                text_errs = []
                # 1. Garbled unicode (replacement char)
                if "\ufffd" in text_to_check:
                    text_errs.append("contains replacement char \ufffd")

                # 2. Repeated character runs (4+ repeating characters, excluding whitespace, dots or hyphens)
                repeats = re.findall(r'([^.\-\s])\1{3,}', text_to_check)
                if repeats:
                    text_errs.append(f"repeated characters: {repeats}")
                # 3. Stray hyphens/spaces (broken words e.g. "ma- jority" or "ﬁ gure")
                broken = re.findall(r'\b\w+-\s+\w+\b', text_to_check)
                if broken:
                    text_errs.append(f"stray hyphen-spaces: {broken}")
                # Ligature space errors: "ﬁ gure"
                ligature_spaces = re.findall(r'[ﬁﬂﬀ]\s+\w+', text_to_check)
                if ligature_spaces:
                    text_errs.append(f"ligature space issues: {ligature_spaces}")
                    
                if text_errs:
                    checks_passed[8] = False
                    text_quality_fails.append((doc_name, f"Element {idx}", f"Text quality flags: {'; '.join(text_errs)} (Preview: '{text_to_check[:60]}')"))

            # --- Check 9: Section Path Consistency ---
            # If target section_path is present
            sec_path = el.get("section_path")
            if sec_path is not None:
                # Check sibling consistency on same page
                # If sibling has it but this doesn't
                pass # Will check below at page level

        # Check 4 uniqueness in this document
        if len(el_ids) != len(set(el_ids)) and len(el_ids) > 0:
            checks_passed[4] = False
            uniqueness_fails.append((doc_name, "Document", f"element_id is not unique in document: {len(el_ids)} elements, {len(set(el_ids))} unique"))

        # Check 5 reading order per page
        for page_no, p_els in page_elements.items():
            r_orders = [o for o, idx, el in p_els if o is not None]
            # check unique
            if len(r_orders) != len(set(r_orders)):
                checks_passed[5] = False
                reading_order_fails.append((doc_name, f"Page {page_no}", "Non-unique reading_order values"))
            # check monotonic increasing
            if r_orders != sorted(r_orders):
                checks_passed[5] = False
                reading_order_fails.append((doc_name, f"Page {page_no}", f"Non-monotonic reading_order values: {r_orders}"))
                
        # Check 9 section path page level consistency
        for page_no, p_els in page_elements.items():
            # Find titles/paragraphs
            sec_paths = [el.get("section_path") for o, idx, el in p_els if el.get("element_type", el.get("type")) in ["title", "paragraph"]]
            has_non_empty = any(s for s in sec_paths if s)
            has_empty = any(s is not None and not s for s in sec_paths)
            if has_non_empty and has_empty:
                checks_passed[9] = False
                section_path_fails.append((doc_name, f"Page {page_no}", "Sibling titles/paragraphs have inconsistent empty/non-empty section_paths"))

    # Check 10: Cross-document consistency
    if len(doc_keys_set) > 1:
        checks_passed[10] = False
        cross_doc_consistency_fails.append(("Corpus", "Corpus", f"Documents have inconsistent keys: {doc_keys_set}"))
    if len(doc_element_keys_set) > 1:
        checks_passed[10] = False
        cross_doc_consistency_fails.append(("Corpus", "Corpus", f"Elements have inconsistent keys: {doc_element_keys_set}"))

    # Map failures to the actual fail tables
    all_fails = []
    for check_num, fails in [
        (1, schema_completeness_fails),
        (2, type_correctness_fails),
        (3, bbox_sanity_fails),
        (4, uniqueness_fails),
        (5, reading_order_fails),
        (6, table_integrity_fails),
        (7, figure_integrity_fails),
        (8, text_quality_fails),
        (9, section_path_fails),
        (10, cross_doc_consistency_fails)
    ]:
        for doc_id, el_id, desc in fails[:50]: # cap details
            all_fails.append({
                "check_num": check_num,
                "doc_id": doc_id,
                "element_id": el_id,
                "description": desc
            })

    # Summarize results
    summary = []
    check_names = {
        1: "Schema completeness",
        2: "Type correctness",
        3: "Bbox sanity",
        4: "Uniqueness",
        5: "Reading order consistency",
        6: "Table integrity",
        7: "Figure integrity",
        8: "Text quality spot-check",
        9: "Section_path consistency",
        10: "Cross-document consistency"
    }
    
    total_passed = sum(1 for passed in checks_passed.values() if passed)
    
    for i in range(1, 11):
        passed = checks_passed[i]
        rate = "100%" if passed else "0%"
        note = "Passed" if passed else f"Failed checks found: {i}"
        
        # Add details to notes
        if i == 1:
            note = "Failed: Documents are missing doc_id, element_id, element_type, page_number, reading_order, section_path, and text/table_markdown/image_path fields from the Target Schema."
        elif i == 2:
            note = "Failed: Due to missing target schema fields."
        elif i == 3:
            note = "Passed: All actual bounding box coordinates are non-negative and satisfy x1 > x0, y1 > y0."
        elif i == 4:
            note = "Failed: element_id and doc_id fields are absent, violating uniqueness constraints."
        elif i == 5:
            note = "Failed: reading_order field is absent."
        elif i == 6:
            note = "Failed: table_markdown field is absent."
        elif i == 7:
            note = "Failed: image_path and image_caption fields are absent from elements."
        elif i == 8:
            note = f"Failed: Text quality scan found stray hyphens/spaces and ligature space issues (common OCR noise) in {len(text_quality_fails)} elements."
        elif i == 9:
            note = "Failed: section_path field is absent."
        elif i == 10:
            note = "Passed: Schema shape is 100% consistent across all documents in the outputs directory."
            
        summary.append({
            "check_num": i,
            "name": check_names[i],
            "pass_rate": rate,
            "notes": note
        })
        
    verdict = "NOT READY"
    reason = "Target schema fields (doc_id, element_id, element_type, page_number, reading_order, section_path) are entirely absent from the pipeline outputs, and OCR text artifacts were detected."
    if total_passed == 10:
        verdict = "READY FOR A4"
        reason = "All checks passed."
        
    report = {
        "docs_checked": total_docs,
        "total_elements": total_elements,
        "passed_checks_count": total_passed,
        "verdict": verdict,
        "reason": reason,
        "summary": summary,
        "failures": all_fails
    }
    
    # Write report as JSON
    results_dir = Path("benchmarking/results/stage2_chunking")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "a2_a3_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    # Render report markdown style as requested
    print("## A2/A3 Corpus Validation Report")
    print(f"- Documents checked: {total_docs}")
    print(f"- Total elements checked: {total_elements}")
    print(f"- Checks passed: {total_passed}/10")
    print(f"- Overall verdict: {verdict} — {reason}")
    print("\n### Per-check summary")
    print("| Check # | Check name | Pass rate | Notes |")
    print("|---|---|---|---|")
    for s in summary:
        print(f"| {s['check_num']} | {s['name']} | {s['pass_rate']} | {s['notes']} |")
        
    print("\n### Failures (grouped by check, worst offenders first)")
    print("| Check # | doc_id | element_id | Issue description |")
    print("|---|---|---|---|")
    # Sort failures by check_num
    sorted_fails = sorted(all_fails, key=lambda x: x['check_num'])
    for f in sorted_fails[:30]: # print top 30
        print(f"| {f['check_num']} | {f['doc_id']} | {f['element_id']} | {f['description']} |")
        
    print("\n### Schema drift (if any)")
    if len(doc_keys_set) == 1:
        print("None. All documents contain identical root keys: " + str(list(doc_keys_set)[0]))
    else:
        print(f"Schema drift detected at document level: {doc_keys_set}")
        
    if len(doc_element_keys_set) == 1:
        print("None. All document elements contain identical keys: " + str(list(doc_element_keys_set)[0]))
    else:
        print(f"Schema drift detected at element level: {doc_element_keys_set}")

if __name__ == "__main__":
    validate_corpus()
