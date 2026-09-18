"""
Strategy F: Semantic Mesh Chunking.
-----------------------------------
A layout-aware multi-modal semantic graph chunker:
1. Phase 1 (Ingestion & De-pagination):
   - Normalizes elements (Text, Table, Figure).
   - Detects and stitches text continuity across page breaks (handling hyphenation
     and sentence continuation while ignoring running headers/footers).
2. Phase 2 (Adaptive Chunking):
   - Text: LLM single-pass ID clustering (Gemini 3.6 Flash) with ~400-750 token
     sweet spot and linked-chain pointers (context_group_id, prev/next pointers).
     Falls back to local topic-drift if API is offline.
   - Tables: Adaptive sizing. Tables <= 35 rows remain 100% atomic (full Markdown
     with headers/captions). Tables > 35 rows produce parent schema chunks and
     header-injected row chunks with summary row pointers.
   - Figures: Strictly atomic (Gemini 3.6 Flash master prompt transcription + visual data).
3. Phase 3 (Post-Chunking Relational Resolution):
   - Scans text chunks for explicit citations ("Table X", "Figure Y") and resolves
     bidirectional pointers between final chunk IDs.
   - Establishes co-page spatial cross-references.
"""
from __future__ import annotations

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from chunking.base import BaseChunker
from chunking.schema import Chunk, union_bbox
from chunking.tokenizer_utils import count_tokens
from chunking.element_types import (
    HEADER_TYPES, TABLE_TYPES, FIGURE_TYPES, FORMULA_TYPES, CAPTION_TYPES, IGNORED_TYPES
)
from chunking.geometry_utils import bbox_gap

logger = logging.getLogger(__name__)

TARGET_MIN_TOKENS = 400
TARGET_MAX_TOKENS = 750
MAX_ATOMIC_TABLE_ROWS = 35


class SemanticMeshChunker(BaseChunker):
    name = "semantic_mesh"

    def __init__(
        self,
        target_min_tokens: int = TARGET_MIN_TOKENS,
        target_max_tokens: int = TARGET_MAX_TOKENS,
        max_atomic_table_rows: int = MAX_ATOMIC_TABLE_ROWS,
        use_gemini_grouper: bool = True,
    ):
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens
        self.max_atomic_table_rows = max_atomic_table_rows
        self.use_gemini_grouper = use_gemini_grouper

    def chunk(self, stage1_json: dict) -> list[Chunk]:
        metadata = stage1_json.get("metadata", {})
        doc_filename = metadata.get("filename", "unknown_doc.pdf")
        doc_stem = doc_filename.rsplit(".", 1)[0]
        elements = stage1_json.get("elements", [])

        if not elements:
            return []

        # -------------------------------------------------------------
        # Phase 1: Ingestion, De-pagination, and Grouping by Page/Section
        # -------------------------------------------------------------
        pages: dict[int, list[tuple[int, dict]]] = {}
        for idx, el in enumerate(elements):
            pages.setdefault(el.get("page", 1), []).append((idx, el))

        all_chunks: list[Chunk] = []
        ignored_elements_by_page: dict[int, list[tuple[int, dict]]] = {}

        # First pass: collect structured sections and non-ignored elements
        current_header_text = "Document General Context"
        current_header_idx = None
        doc_sections: list[dict] = []

        for page in sorted(pages.keys()):
            page_elements = pages[page]
            non_ignored = []
            ignored = []
            for idx, el in page_elements:
                el_type = el.get("type", "plain text")
                if el_type in IGNORED_TYPES or el.get("element_type") == "boilerplate":
                    ignored.append((idx, el))
                else:
                    non_ignored.append((idx, el))
            ignored_elements_by_page[page] = ignored

            local_header_types = HEADER_TYPES | {"figure_caption", "table_caption"}
            for idx, el in non_ignored:
                el_type = el.get("type", "plain text")
                if el_type in local_header_types:
                    content = el.get("content", "").strip()
                    if content:
                        current_header_text = content
                        current_header_idx = idx

                doc_sections.append({
                    "idx": idx,
                    "el": el,
                    "page": page,
                    "header_text": current_header_text,
                    "header_idx": current_header_idx,
                })

        # -------------------------------------------------------------
        # Phase 2: Adaptive Chunking (Text, Tables, Figures)
        # -------------------------------------------------------------
        raw_chunks: list[Chunk] = []
        seq = 0

        # Process elements in sequential flow
        i = 0
        total_items = len(doc_sections)

        while i < total_items:
            item = doc_sections[i]
            idx = item["idx"]
            el = item["el"]
            page = item["page"]
            header_text = item["header_text"]
            el_type = el.get("type", "plain text")

            # --- TABLE PROCESSING ---
            if el_type in TABLE_TYPES:
                table_extracted = el.get("extracted") if isinstance(el.get("extracted"), dict) else {}
                table_md = table_extracted.get("markdown") or el.get("content", "")
                table_obj = table_extracted.get("table")
                rows = self._parse_table_rows(table_md, table_obj)
                bbox = el.get("bbox", [0.0, 0.0, 0.0, 0.0])
                table_id = self._make_chunk_id(doc_stem, page, seq)

                # Adaptive Sizing Decision
                if len(rows) <= self.max_atomic_table_rows or len(rows) < 2:
                    # Atomic Table Chunk
                    caption = el.get("content") or ""
                    header_prefix = f"{header_text}\n\n{caption}" if (caption and caption != header_text and not caption.startswith("|")) else header_text
                    table_text = f"{header_prefix}\n\n{table_md}" if header_prefix else table_md
                    raw_chunks.append(Chunk(
                        chunk_id=table_id,
                        doc_filename=doc_filename,
                        page=page,
                        strategy=self.name,
                        element_types=[el_type],
                        bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        text=table_text,
                        token_count=count_tokens(table_text),
                        parent_section=header_text,
                        source_element_indices=[idx],
                        page_span=[page],
                        table_pointers={"is_atomic": True, "total_rows": len(rows)},
                        metadata={"caption": el.get("content") or header_text, "source": "table_atomic"},
                    ))
                    seq += 1
                else:
                    # Large Table: Emit Parent Schema Chunk + Header-Injected Row Chunks
                    headers = rows[0]
                    data_rows = rows[1:]
                    caption = el.get("content") or header_text

                    parent_chunk_id = f"{table_id}::parent"
                    parent_text = f"Table: {caption} | Columns: {', '.join(str(h) for h in headers)} | Rows: {len(data_rows)}"
                    raw_chunks.append(Chunk(
                        chunk_id=parent_chunk_id,
                        doc_filename=doc_filename,
                        page=page,
                        strategy=self.name,
                        element_types=[el_type],
                        bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        text=parent_text,
                        token_count=count_tokens(parent_text),
                        parent_section=header_text,
                        source_element_indices=[idx],  # parent accounts for source index
                        page_span=[page],
                        table_pointers={"is_parent": True, "total_rows": len(data_rows)},
                        metadata={"caption": caption, "source": "table_parent"},
                    ))

                    # Emit Child Row Chunks with Linked Pointers
                    from benchmark_harness.stages.table_header_injection import build_row_chunks
                    row_chunks = build_row_chunks(
                        table_id=table_id,
                        headers=[str(h) for h in headers],
                        rows=data_rows,
                        caption=caption,
                        section_title=header_text,
                        fiscal_period_col=0,
                    )
                    for r_chunk in row_chunks:
                        raw_chunks.append(Chunk(
                            chunk_id=r_chunk.chunk_id,
                            doc_filename=doc_filename,
                            page=page,
                            strategy=self.name,
                            element_types=[el_type],
                            bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                            text=r_chunk.text,
                            token_count=count_tokens(r_chunk.text),
                            parent_section=header_text,
                            source_element_indices=[],  # child rows link to parent
                            page_span=[page],
                            table_pointers=r_chunk.metadata.get("table_pointers", {}),
                            metadata=r_chunk.metadata,
                        ))
                    seq += 1

                i += 1
                continue

            # --- FIGURE / IMAGE PROCESSING ---
            if el_type in FIGURE_TYPES:
                bbox = el.get("bbox", [0.0, 0.0, 0.0, 0.0])
                fig_content = el.get("content", "") or ""
                fig_text = f"{header_text}\n\n[FIGURE] {fig_content}" if header_text else f"[FIGURE] {fig_content}"
                raw_chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=doc_filename,
                    page=page,
                    strategy=self.name,
                    element_types=[el_type],
                    bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    text=fig_text,
                    token_count=count_tokens(fig_text),
                    parent_section=header_text,
                    source_element_indices=[idx],
                    page_span=[page],
                    metadata={"figure_caption": fig_content, "image_path": el.get("image_path")},
                ))
                seq += 1
                i += 1
                continue

            # --- FORMULA / CAPTION PROCESSING ---
            if el_type in FORMULA_TYPES or el_type in CAPTION_TYPES:
                bbox = el.get("bbox", [0.0, 0.0, 0.0, 0.0])
                content = el.get("content", "") or ""
                raw_chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=doc_filename,
                    page=page,
                    strategy=self.name,
                    element_types=[el_type],
                    bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    text=content,
                    token_count=count_tokens(content),
                    parent_section=header_text,
                    source_element_indices=[idx],
                    page_span=[page],
                    metadata={"element_type": el_type},
                ))
                seq += 1
                i += 1
                continue

            # --- PROSE TEXT PROCESSING (with Cross-Page Stitching & Grouping) ---
            # Collect contiguous block of text-like elements under current section
            text_cluster = []
            cluster_header = header_text
            while i < total_items:
                next_item = doc_sections[i]
                next_el_type = next_item["el"].get("type", "plain text")
                if next_el_type in TABLE_TYPES | FIGURE_TYPES | FORMULA_TYPES:
                    break
                if next_item["header_text"] != cluster_header and text_cluster:
                    # New section header encountered
                    break
                text_cluster.append(next_item)
                i += 1

            if not text_cluster:
                i += 1
                continue

            # Process the text cluster into semantic chunks
            section_slug = re.sub(r'[^a-zA-Z0-9]', '_', cluster_header[:30]).lower().strip('_') or "sec"
            context_group_id = f"{doc_stem}_{section_slug}_{seq:03d}"
            sub_chunks = self._render_text_cluster(
                text_cluster=text_cluster,
                header_text=cluster_header,
                doc_stem=doc_stem,
                doc_filename=doc_filename,
                context_group_id=context_group_id,
                seq_start=seq,
            )
            raw_chunks.extend(sub_chunks)
            seq += len(sub_chunks)

        # -------------------------------------------------------------
        # Re-attach Ignored Elements to Nearest Chunks (Coverage Guard)
        # -------------------------------------------------------------
        for p, ignored_list in ignored_elements_by_page.items():
            page_chunks = [c for c in raw_chunks if c.page == p]
            for ignored_idx, ignored_el in ignored_list:
                if page_chunks:
                    best_chunk = min(
                        page_chunks,
                        key=lambda c: bbox_gap(ignored_el["bbox"], c.bbox_union)
                    )
                    best_chunk.source_element_indices.append(ignored_idx)
                elif raw_chunks:
                    raw_chunks[0].source_element_indices.append(ignored_idx)

        # -------------------------------------------------------------
        # Phase 3: Post-Chunking Relational Mesh Cross-Referencing
        # -------------------------------------------------------------
        final_chunks = self._resolve_cross_references(raw_chunks)
        return final_chunks

    def _render_text_cluster(
        self,
        text_cluster: list[dict],
        header_text: str,
        doc_stem: str,
        doc_filename: str,
        context_group_id: str,
        seq_start: int,
    ) -> list[Chunk]:
        """Groups contiguous text elements into ~400-750 token chunks with linked-chain pointers."""
        grouped_indices: list[list[int]] = []
        if self.use_gemini_grouper and len(text_cluster) > 1:
            total_tokens = sum(count_tokens(item["el"].get("content", "") or "") for item in text_cluster)
            if total_tokens > self.target_max_tokens:
                grouped_indices = self._gemini_cluster_blocks(text_cluster)

        # Fallback to local sentence/token topic-drift grouping if Gemini not used or failed
        if not grouped_indices:
            grouped_indices = self._local_cluster_blocks(text_cluster)

        chunks: list[Chunk] = []
        total_groups = len(grouped_indices)

        for g_idx, block_group in enumerate(grouped_indices):
            group_items = [text_cluster[bi] for bi in block_group]
            source_indices = [item["idx"] for item in group_items]
            pages_in_group = sorted(list(set(item["page"] for item in group_items)))
            bboxes = [item["el"]["bbox"] for item in group_items]
            el_types = [item["el"].get("type", "plain text") for item in group_items]

            # Cross-page text stitching: join with double-newline, but fix hyphenation if continuing
            text_parts = []
            for item in group_items:
                content = (item["el"].get("content", "") or "").strip()
                if content:
                    text_parts.append(content)

            body = "\n\n".join(text_parts)
            # Fix split hyphens across page transitions: e.g. "disag-\n\nreement" -> "disagreement"
            body = re.sub(r'(\w+)-\s*\n\n\s*(\w+)', r'\1\2', body)

            chunk_text = f"{header_text}\n\n{body}" if header_text and body != header_text else body
            primary_page = pages_in_group[0] if pages_in_group else 1
            chunk_seq = seq_start + g_idx
            chunk_id = self._make_chunk_id(doc_stem, primary_page, chunk_seq)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_filename=doc_filename,
                page=primary_page,
                strategy=self.name,
                element_types=el_types,
                bbox_union=union_bbox(bboxes) if bboxes else (0.0, 0.0, 1.0, 1.0),
                text=chunk_text,
                token_count=count_tokens(chunk_text),
                parent_section=header_text,
                source_element_indices=source_indices,
                page_span=pages_in_group,
                lineage={
                    "context_group_id": context_group_id,
                    "sequence_in_group": g_idx + 1,
                    "total_in_group": total_groups,
                    "prev_chunk_id": None,  # populated in next step
                    "next_chunk_id": None,  # populated in next step
                },
                metadata={"header": header_text},
            ))

        # Wire bidirectional lineage pointers within the context group
        for idx_c in range(len(chunks)):
            if idx_c > 0:
                chunks[idx_c].lineage["prev_chunk_id"] = chunks[idx_c - 1].chunk_id
            if idx_c < len(chunks) - 1:
                chunks[idx_c].lineage["next_chunk_id"] = chunks[idx_c + 1].chunk_id

        return chunks

    def _local_cluster_blocks(self, text_cluster: list[dict]) -> list[list[int]]:
        """Greedy sentence/paragraph-aware grouping respecting ~400-750 token budget."""
        groups: list[list[int]] = []
        current_group: list[int] = []
        current_tokens = 0

        for i, item in enumerate(text_cluster):
            content = item["el"].get("content", "") or ""
            t_count = count_tokens(content)

            if current_group and (current_tokens + t_count > self.target_max_tokens) and current_tokens >= self.target_min_tokens:
                groups.append(current_group)
                current_group = [i]
                current_tokens = t_count
            else:
                current_group.append(i)
                current_tokens += t_count

        if current_group:
            groups.append(current_group)
        return groups

    def _gemini_cluster_blocks(self, text_cluster: list[dict]) -> list[list[int]]:
        """Calls Gemini 3.6 Flash once with block IDs to group them semantically."""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return []

        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            blocks_manifest = []
            for i, item in enumerate(text_cluster):
                c = (item["el"].get("content", "") or "").strip()
                t_cnt = count_tokens(c)
                blocks_manifest.append(f"[B{i:02d}] (page {item['page']}, tokens: {t_cnt}): {c[:160]}")

            prompt = (
                "You are an expert document structural analyst.\n"
                f"We have {len(text_cluster)} consecutive text blocks. Group them into cohesive semantic chunks.\n"
                f"Guidelines:\n"
                f"- Target each chunk between {self.target_min_tokens} and {self.target_max_tokens} tokens.\n"
                "- Keep continuous arguments or narratives in the same group.\n"
                "- Start a new group when there is an abrupt topic shift.\n"
                "- Output ONLY valid JSON: an array of index arrays, covering every block index 0 to "
                f"{len(text_cluster)-1} in ascending order without omitting any. "
                "Example: [[0, 1], [2, 3, 4]]"
            )
            full_content = prompt + "\n\nBlocks:\n" + "\n".join(blocks_manifest)

            resp = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=full_content,
            )
            if resp and resp.text:
                match = re.search(r'\[\s*\[.*?\]\s*\]', resp.text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    all_ids = [idx for grp in parsed for idx in grp]
                    if sorted(all_ids) == list(range(len(text_cluster))):
                        return parsed
        except Exception as e:
            logger.debug(f"[SemanticMesh] Gemini block grouping fallback triggered: {e}")

        return []

    def _parse_table_rows(self, table_md: str, table_obj: Any) -> list[list[Any]]:
        """Extracts rows from either structured table object or markdown string."""
        if isinstance(table_obj, list) and len(table_obj) > 0 and isinstance(table_obj[0], list):
            return table_obj

        rows = []
        if table_md:
            lines = [line.strip() for line in table_md.split("\n") if line.strip().startswith("|")]
            for line in lines:
                if set(line.replace("|", "").strip()) <= {"-", ":"}:
                    continue
                cells = [c.strip() for c in line.strip("|").split("|")]
                if any(cells):
                    rows.append(cells)
        return rows

    def _resolve_cross_references(self, chunks: list[Chunk]) -> list[Chunk]:
        """Post-chunking pass: scans text for Table/Figure citations and builds bidirectional mesh."""
        table_chunks: dict[str, Chunk] = {}
        figure_chunks: dict[str, Chunk] = {}

        for c in chunks:
            if "table" in c.element_types:
                table_chunks[c.chunk_id] = c
            elif "figure" in c.element_types:
                figure_chunks[c.chunk_id] = c

        table_num_pattern = re.compile(r'(?:Table|TABLE)\s+([A-Za-z0-9\.\-_]+)')
        fig_num_pattern = re.compile(r'(?:Figure|FIGURE|Fig\.|Chart|CHART)\s+([A-Za-z0-9\.\-_]+)')

        for c in chunks:
            if not c.cross_references:
                c.cross_references = {
                    "tables": [],
                    "figures": [],
                    "co_page_tables": [],
                    "co_page_figures": [],
                    "co_page_text": [],
                    "cited_by_text": [],
                    "citations_detected": [],
                }

            # 1. Co-page spatial proximity links
            co_page_tables = [tc.chunk_id for tc in table_chunks.values() if tc.page == c.page and tc.chunk_id != c.chunk_id]
            co_page_figures = [fc.chunk_id for fc in figure_chunks.values() if fc.page == c.page and fc.chunk_id != c.chunk_id]
            c.cross_references["co_page_tables"] = co_page_tables
            c.cross_references["co_page_figures"] = co_page_figures

            # 2. Text Citation Detection
            if "plain text" in c.element_types or "title" in c.element_types:
                t_matches = table_num_pattern.findall(c.text)
                f_matches = fig_num_pattern.findall(c.text)

                if t_matches:
                    c.cross_references["citations_detected"].extend([f"Table {m}" for m in t_matches])
                    for tc_id, tc in table_chunks.items():
                        if tc.page == c.page or any(m in tc.text for m in t_matches):
                            if tc_id not in c.cross_references["tables"]:
                                c.cross_references["tables"].append(tc_id)
                                if not tc.cross_references:
                                    tc.cross_references = {"cited_by_text": [], "tables": [], "figures": [], "co_page_tables": [], "co_page_figures": [], "co_page_text": [], "citations_detected": []}
                                if c.chunk_id not in tc.cross_references["cited_by_text"]:
                                    tc.cross_references["cited_by_text"].append(c.chunk_id)

                if f_matches:
                    c.cross_references["citations_detected"].extend([f"Figure {m}" for m in f_matches])
                    for fc_id, fc in figure_chunks.items():
                        if fc.page == c.page or any(m in fc.text for m in f_matches):
                            if fc_id not in c.cross_references["figures"]:
                                c.cross_references["figures"].append(fc_id)
                                if not fc.cross_references:
                                    fc.cross_references = {"cited_by_text": [], "tables": [], "figures": [], "co_page_tables": [], "co_page_figures": [], "co_page_text": [], "citations_detected": []}
                                if c.chunk_id not in fc.cross_references["cited_by_text"]:
                                    fc.cross_references["cited_by_text"].append(c.chunk_id)

        return chunks
