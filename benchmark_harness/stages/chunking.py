from typing import List, Dict, Any


def get_chunker(strategy_name: str = "semantic_mesh"):
    if strategy_name == "semantic_mesh":
        from chunking.strategies.semantic_mesh import SemanticMeshChunker
        return SemanticMeshChunker()
    elif strategy_name == "section_hierarchical":
        from chunking.strategies.section_hierarchical import SectionHierarchicalChunker
        return SectionHierarchicalChunker()
    elif strategy_name == "hybrid_semantic":
        from chunking.strategies.hybrid_semantic import HybridSemanticChunker
        return HybridSemanticChunker()
    else:
        from chunking.strategies.semantic_mesh import SemanticMeshChunker
        return SemanticMeshChunker()


def chunk_document(doc_text_or_json: Any, strategy_name: str = "semantic_mesh") -> List[Dict[str, Any]]:
    """
    Chunks document (stage1_json dict or plain text) into a list of chunk dicts:
    [{ "chunk_id": str, "text": str, "page_span": list, "lineage": dict, "cross_references": dict, ... }]
    """
    if isinstance(doc_text_or_json, dict) and "elements" in doc_text_or_json:
        chunker = get_chunker(strategy_name)
        chunks = chunker.chunk(doc_text_or_json)
        return [c.to_dict() for c in chunks]

    # Tag-aware string chunking (parses [TABLE], [FIGURE], [TEXT] tags with cross-references)
    import re
    text = str(doc_text_or_json)
    if not text.strip():
        return []

    # Regex patterns for tags
    tag_pattern = re.compile(r'\[(FIGURE|TABLE|TEXT)\s+id="([^"]+)"\s+page="([^"]+)"(.*?)\]')
    
    # Split on double newline (blocks)
    raw_blocks = [p.strip() for p in text.split("\n\n") if p.strip()]
    res_chunks = []
    curr_parts = []
    curr_len = 0
    c_idx = 0

    def flush_text_chunk():
        nonlocal curr_parts, curr_len, c_idx, res_chunks
        if not curr_parts:
            return
        chunk_body = "\n\n".join(curr_parts)
        res_chunks.append({
            "chunk_id": f"chunk_{c_idx}",
            "text": chunk_body,
            "element_types": ["text"],
            "lineage": {
                "context_group_id": "text_stream",
                "sequence_in_group": c_idx + 1,
                "prev_chunk_id": f"chunk_{c_idx - 1}" if c_idx > 0 else None,
                "next_chunk_id": None,
            },
            "cross_references": {},
        })
        c_idx += 1
        curr_parts = []
        curr_len = 0

    for block in raw_blocks:
        tag_match = tag_pattern.search(block)
        is_table = "[TABLE]" in block or (tag_match and tag_match.group(1) == "TABLE")
        is_figure = "[FIGURE" in block or (tag_match and tag_match.group(1) == "FIGURE")

        if is_table or is_figure:
            flush_text_chunk()
            elem_type = "table" if is_table else "figure"
            xref = {}
            if tag_match:
                tag_kind, el_id, page_num, extra_attrs = tag_match.groups()
                xref["element_id"] = el_id
                xref["page"] = page_num
                # Extract cross-ref attributes
                for attr_match in re.finditer(r'cross_ref_(\w+)="([^"]*)"', extra_attrs):
                    attr_name, attr_val = attr_match.groups()
                    xref[f"cross_ref_{attr_name}"] = [v.strip() for v in attr_val.split(",") if v.strip()]

            res_chunks.append({
                "chunk_id": f"chunk_{c_idx}",
                "text": block,
                "element_types": [elem_type],
                "lineage": {
                    "context_group_id": f"{elem_type}_atomic",
                    "sequence_in_group": 1,
                    "prev_chunk_id": f"chunk_{c_idx - 1}" if c_idx > 0 else None,
                    "next_chunk_id": None,
                },
                "cross_references": xref,
            })
            c_idx += 1
        else:
            if curr_parts and (curr_len + len(block) > 2000):
                flush_text_chunk()
            curr_parts.append(block)
            curr_len += len(block)

    flush_text_chunk()

    # Wire forward pointers across all chunks
    for i in range(len(res_chunks) - 1):
        res_chunks[i]["lineage"]["next_chunk_id"] = res_chunks[i + 1]["chunk_id"]

    return res_chunks

