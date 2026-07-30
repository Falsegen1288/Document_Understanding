"""Strategy C: Section-Hierarchical Chunking.
Uses section_header/title elements as hard chunk boundaries. All elements between one
header and the next belong to that section. The header text is prepended to every
chunk derived from that section (contextual chunk pattern). Long sections are
sub-split by paragraph while keeping the header prefix. Tables, figures, formulas,
and captions/footnotes remain atomic and carry a `parent_section` tag."""
import logging
from chunking.base import BaseChunker
from chunking.schema import Chunk, union_bbox
from chunking.tokenizer_utils import count_tokens
from chunking.element_types import HEADER_TYPES, TABLE_TYPES, FIGURE_TYPES, FORMULA_TYPES, CAPTION_TYPES, IGNORED_TYPES

logger = logging.getLogger(__name__)

MAX_SECTION_TOKENS = 800  # sub-split threshold; tune after seeing real distributions


class SectionHierarchicalChunker(BaseChunker):
    name = "section_hierarchical"

    def __init__(self, max_section_tokens: int = MAX_SECTION_TOKENS):
        self.max_section_tokens = max_section_tokens

    def chunk(self, stage1_json: dict) -> list[Chunk]:
        doc_stem = stage1_json["metadata"]["filename"].rsplit(".", 1)[0]
        doc_filename = stage1_json["metadata"]["filename"]
        elements = stage1_json["elements"]

        pages: dict[int, list[tuple[int, dict]]] = {}
        for idx, el in enumerate(elements):
            pages.setdefault(el["page"], []).append((idx, el))

        all_chunks: list[Chunk] = []
        current_header_text = "Untitled Section"
        current_header_idx = None

        for page in sorted(pages.keys()):
            page_elements = pages[page]
            
            # Filter non-ignored and ignored elements on this page
            non_ignored = [(idx, el) for idx, el in page_elements if el["type"] not in IGNORED_TYPES]
            ignored = [(idx, el) for idx, el in page_elements if el["type"] in IGNORED_TYPES]
            
            sections = self._split_into_sections(non_ignored, current_header_text, current_header_idx)

            page_chunks = []
            for header_text, header_idx, group in sections:
                current_header_text = header_text
                current_header_idx = header_idx
                seq_base = len(all_chunks) + len(page_chunks)
                sub_chunks = self._render_section(
                    group, header_text, doc_stem, page, seq_base, doc_filename, elements
                )
                page_chunks.extend(sub_chunks)

            # Attach ignored elements to the nearest chunk on this page
            for ignored_idx, ignored_el in ignored:
                if page_chunks:
                    from chunking.geometry_utils import bbox_gap
                    best_chunk = min(
                        page_chunks,
                        key=lambda c: bbox_gap(ignored_el["bbox"], c.bbox_union)
                    )
                    best_chunk.source_element_indices.append(ignored_idx)

            all_chunks.extend(page_chunks)
        return all_chunks

    def _split_into_sections(self, page_elements, carry_header_text, carry_header_idx):
        """Split one page's elements into (header_text, header_idx, [elements])
        groups. If the page starts before any header is seen, the carried-over
        header from the previous page is used."""
        sections = []
        current_group = []
        header_text = carry_header_text
        header_idx = carry_header_idx

        local_header_types = HEADER_TYPES | {"figure_caption", "table_caption"}
        for idx, el in page_elements:
            if el["type"] in local_header_types:
                if current_group:
                    sections.append((header_text, header_idx, current_group))
                header_text = el.get("content", "").strip()
                header_idx = idx
                current_group = [(idx, el)]  # header itself is part of its own section
            else:
                current_group.append((idx, el))

        if current_group:
            sections.append((header_text, header_idx, current_group))
        return sections

    def _render_section(self, group, header_text, doc_stem, page, seq_base, doc_filename, elements):
        """Render one section's elements into one or more Chunks, sub-splitting on
        paragraph boundaries if the section exceeds max_section_tokens. Tables,
        figures, formulas, and captions/footnotes remain atomic chunks."""
        chunks = []
        running_text_parts = []
        running_indices = []
        running_bboxes = []
        seq = seq_base

        # Keep track of preceding table/figure/formula in the same section group for caption matching
        last_anchor_type = None

        def flush():
            nonlocal running_text_parts, running_indices, running_bboxes, seq
            if not running_text_parts:
                return
            body = "\n\n".join(running_text_parts)
            text = f"{header_text}\n\n{body}" if body.strip() != header_text.strip() else header_text
            element_types = [elements[i]["type"] for i in running_indices]
            chunks.append(Chunk(
                chunk_id=self._make_chunk_id(doc_stem, page, seq),
                doc_filename=doc_filename,
                page=page,
                strategy=self.name,
                element_types=element_types,
                bbox_union=union_bbox(running_bboxes),
                text=text,
                token_count=count_tokens(text),
                parent_section=header_text,
                source_element_indices=list(running_indices),
            ))
            seq += 1
            running_text_parts, running_indices, running_bboxes = [], [], []

        for idx, el in group:
            el_type = el["type"]
            if el_type in TABLE_TYPES:
                flush()
                last_anchor_type = "table"
                bbox = el["bbox"]
                table_md = el["extracted"].get("markdown", "") if el.get("extracted") is not None else el.get("content", "")
                page_elements = [(i, e) for i, e in enumerate(elements) if e["page"] == page]
                aligned_header = self._find_aligned_header(page_elements, bbox)
                header_context = aligned_header if aligned_header else header_text
                text = f"{header_context}\n\n{table_md}" if header_context else table_md
                chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=doc_filename,
                    page=page,
                    strategy=self.name,
                    element_types=[el_type],
                    bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    text=text,
                    token_count=count_tokens(text),
                    parent_section=header_context,
                    source_element_indices=[idx],
                    metadata={"table_extractor": el["extracted"].get("extractor")} if el.get("extracted") is not None else {},
                ))
                seq += 1
            elif el_type in FIGURE_TYPES:
                flush()
                last_anchor_type = "figure"
                bbox = el["bbox"]
                fig_content = el.get('content', '') or ""
                page_elements = [(i, e) for i, e in enumerate(elements) if e["page"] == page]
                aligned_header = self._find_aligned_header(page_elements, bbox)
                header_context = aligned_header if aligned_header else header_text
                text = f"{header_context}\n\n[FIGURE] {fig_content}" if header_context else f"[FIGURE] {fig_content}"
                chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=doc_filename,
                    page=page,
                    strategy=self.name,
                    element_types=[el_type],
                    bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    text=text,
                    token_count=count_tokens(text),
                    parent_section=header_context,
                    source_element_indices=[idx],
                ))
                seq += 1
            elif el_type in FORMULA_TYPES:
                flush()
                last_anchor_type = "formula"
                bbox = el["bbox"]
                text = el.get("content", "") or ""
                chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=doc_filename,
                    page=page,
                    strategy=self.name,
                    element_types=[el_type],
                    bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    text=text,
                    token_count=count_tokens(text),
                    parent_section=header_text,
                    source_element_indices=[idx],
                ))
                seq += 1
            elif el_type in CAPTION_TYPES:
                flush()
                bbox = el["bbox"]
                text = el.get("content", "") or ""
                # Find caption association
                caption_metadata = {}
                if last_anchor_type is not None:
                    caption_metadata["caption_for"] = last_anchor_type
                
                chunks.append(Chunk(
                    chunk_id=self._make_chunk_id(doc_stem, page, seq),
                    doc_filename=doc_filename,
                    page=page,
                    strategy=self.name,
                    element_types=[el_type],
                    bbox_union=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    text=text,
                    token_count=count_tokens(text),
                    parent_section=header_text,
                    source_element_indices=[idx],
                    metadata=caption_metadata,
                ))
                seq += 1
            else:
                content = el.get("content", "") or ""
                prospective_tokens = count_tokens("\n\n".join(running_text_parts + [content]))
                if prospective_tokens > self.max_section_tokens and running_text_parts:
                    flush()
                running_text_parts.append(content)
                running_indices.append(idx)
                running_bboxes.append(el["bbox"])

        flush()
        return chunks

    def _find_aligned_header(self, page_elements, anchor_bbox):
        local_header_types = {"title", "figure_caption"}
        best_gap = 99999.0
        best_text = None
        for idx, el in page_elements:
            if el["type"] in local_header_types:
                # Check if the header starts above the table
                if el["bbox"][3] <= anchor_bbox[1] + 15:
                    gap = anchor_bbox[1] - el["bbox"][3]
                    if gap < best_gap:
                        best_gap = gap
                        best_text = el.get("content", "").strip()
        return best_text

