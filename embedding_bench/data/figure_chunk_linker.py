import json
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class LinkedChunk:
    chunk_id: str
    text: str
    figure_image_path: str | None
    link_confidence: float

def get_center(bbox):
    if not bbox or len(bbox) < 4:
        return (0.0, 0.0)
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

def get_distance(bbox1, bbox2):
    c1 = get_center(bbox1)
    c2 = get_center(bbox2)
    return float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

def build_linked_chunk_set(
    corpus_dir: Path,
    output_path: Path,
    min_confidence: float = 0.5
) -> list[LinkedChunk]:
    doc_stems = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]
    linked_chunks = []

    for stem in doc_stems:
        chunks_json = corpus_dir / f"{stem}_hybrid_semantic.json"
        if not chunks_json.exists():
            # Try outputs/chunks/
            chunks_json = Path("outputs/chunks") / f"{stem}_hybrid_semantic.json"
            if not chunks_json.exists():
                raise FileNotFoundError(f"Chunks file not found for {stem}")
                
        layout_json = corpus_dir.parent / stem / f"{stem}.json"
        if not layout_json.exists():
            layout_json = corpus_dir / stem / f"{stem}.json"
        if not layout_json.exists():
            layout_json = Path("outputs") / stem / f"{stem}.json"
            if not layout_json.exists():
                raise FileNotFoundError(f"Layout file not found for {stem}")

        with open(chunks_json, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        with open(layout_json, "r", encoding="utf-8") as f:
            layout_data = json.load(f)
            
        elements = layout_data.get("elements", [])
        
        # Group figure elements by page
        figures_by_page = {}
        for idx, el in enumerate(elements):
            if el.get("type") == "figure" or el.get("element_type") == "figure":
                img_path = el.get("image_path")
                # Normalize path separators
                if img_path:
                    img_path = img_path.replace("\\", "/")
                if img_path and os.path.exists(img_path):
                    page = el.get("page") or el.get("page_number")
                    if page not in figures_by_page:
                        figures_by_page[page] = []
                    figures_by_page[page].append((idx, el))

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk.get("text", "")
            page = chunk.get("page")
            
            # Check if this chunk is a table or contains a table
            element_types = chunk.get("element_types", [])
            is_table_chunk = any(t == "table" for t in element_types) or "table" in text.lower() or "figure" in text.lower()

            best_img_path = None
            best_conf = 0.0
            
            if is_table_chunk and page in figures_by_page:
                chunk_bbox = chunk.get("bbox_union")
                source_indices = chunk.get("source_element_indices", [])
                
                # Check horizontal/vertical alignment
                candidates = []
                for fig_idx, fig_el in figures_by_page[page]:
                    fig_bbox = fig_el.get("bbox")
                    fig_path = fig_el.get("image_path").replace("\\", "/")
                    
                    dist = get_distance(chunk_bbox, fig_bbox)
                    
                    # Column alignment: check vertical centers overlap or close horizontal alignment
                    c_chunk = get_center(chunk_bbox)
                    c_fig = get_center(fig_bbox)
                    v_aligned = abs(c_chunk[1] - c_fig[1]) < 100.0
                    
                    # Reading order alignment
                    r_aligned = False
                    for src_idx in source_indices:
                        if src_idx < len(elements):
                            if abs(elements[src_idx].get("reading_order", 0) - fig_el.get("reading_order", 0)) <= 3:
                                r_aligned = True
                                break
                                
                    if v_aligned or r_aligned:
                        conf = 1.0
                    else:
                        conf = 0.5
                        
                    candidates.append((conf, dist, fig_path))
                    
                if candidates:
                    # Sort by confidence descending, then distance ascending
                    candidates.sort(key=lambda x: (-x[0], x[1]))
                    top_conf, _, top_path = candidates[0]
                    if top_conf >= min_confidence:
                        best_img_path = top_path
                        best_conf = top_conf
                        
            linked_chunks.append(LinkedChunk(
                chunk_id=chunk_id,
                text=text,
                figure_image_path=best_img_path,
                link_confidence=best_conf
            ))

    # Write JSONL output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for lc in linked_chunks:
            f.write(json.dumps(asdict(lc)) + "\n")
            
    return linked_chunks
