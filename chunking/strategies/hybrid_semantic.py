"""Strategy E: Hybrid Semantic-Structural Chunking.
Inherits Strategy C's hard structural boundaries (section headers, atomic
tables/figures/captions/formulas) unchanged. Within a section's running text-like
elements, adds a semantic soft-split: consecutive elements whose embeddings show a
topic shift (low cosine similarity) get split into separate sub-chunks even if the
token budget hasn't been reached yet. Structural boundaries always take precedence
over semantic ones — this pass only refines splits WITHIN what Strategy C already
groups into one section."""
import logging

from chunking.schema import Chunk, union_bbox
from chunking.tokenizer_utils import count_tokens
from chunking.embedding_utils import get_default_embedder, cosine_similarity
from chunking.element_types import TABLE_TYPES, FIGURE_TYPES, FORMULA_TYPES, CAPTION_TYPES, IGNORED_TYPES
from chunking.strategies.section_hierarchical import SectionHierarchicalChunker, MAX_SECTION_TOKENS

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.5   # below this, consecutive elements are considered a topic shift
MIN_CHUNK_TOKENS = 80        # floor before a semantic split is allowed to fire

ATOMIC_IN_SECTION = TABLE_TYPES | FIGURE_TYPES | FORMULA_TYPES | CAPTION_TYPES


class HybridSemanticChunker(SectionHierarchicalChunker):
    name = "hybrid_semantic"

    def __init__(
        self,
        max_section_tokens: int = MAX_SECTION_TOKENS,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    ):
        super().__init__(max_section_tokens=max_section_tokens)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_tokens = min_chunk_tokens
        self.embedder = get_default_embedder()
        self.semantic_flushes = 0
        self.token_flushes = 0

    def chunk(self, stage1_json: dict) -> list[Chunk]:
        # Reset counters for the new document run
        self.semantic_flushes = 0
        self.token_flushes = 0
        
        chunks = super().chunk(stage1_json)
        
        logger.info(
            f"[{self.name}] Completed chunking. "
            f"Semantic flushes: {self.semantic_flushes}, Token flushes: {self.token_flushes}"
        )
        return chunks

    def _render_section(self, group, header_text, doc_stem, page, seq_base, doc_filename, elements):
        """Overrides Strategy C's renderer. Atomic-type handling (table/figure/
        formula/caption) is copied unchanged from the parent; only the running
        text-like accumulation loop differs, using semantic similarity instead of
        pure token-count as the flush trigger."""
        chunks = []
        seq = seq_base

        # Pull out just the text-like elements in this group to batch-embed once
        # (avoids one embedding call per element during the sequential walk).
        text_like = [(idx, el) for idx, el in group if el["type"] not in ATOMIC_IN_SECTION]
        text_contents = [el.get("content", "") or "" for _, el in text_like]
        embeddings = self.embedder.embed(text_contents) if text_contents else None
        embed_lookup = {
            idx: embeddings[i] for i, (idx, _) in enumerate(text_like)
        } if embeddings is not None else {}

        running_text_parts, running_indices, running_bboxes = [], [], []
        last_embedding = None

        def flush():
            nonlocal running_text_parts, running_indices, running_bboxes, seq, last_embedding
            if not running_text_parts:
                return
            body = "\n\n".join(running_text_parts)
            text = f"{header_text}\n\n{body}" if body.strip() != header_text.strip() else header_text
            chunks.append(Chunk(
                chunk_id=self._make_chunk_id(doc_stem, page, seq),
                doc_filename=doc_filename,
                page=page,
                strategy=self.name,
                element_types=[group_type_lookup[i] for i in running_indices],
                bbox_union=union_bbox(running_bboxes),
                text=text,
                token_count=count_tokens(text),
                parent_section=header_text,
                source_element_indices=list(running_indices),
            ))
            seq += 1
            running_text_parts, running_indices, running_bboxes = [], [], []
            last_embedding = None

        group_type_lookup = {idx: el["type"] for idx, el in group}

        # Keep track of preceding table/figure/formula in the same section group for caption matching
        last_anchor_type = None

        for idx, el in group:
            el_type = el["type"]
            if el_type in ATOMIC_IN_SECTION:
                flush()  # close any pending running text before the atomic element
                bbox = el["bbox"]
                if el_type in TABLE_TYPES:
                    last_anchor_type = "table"
                    text = el["extracted"].get("markdown", "") if el.get("extracted") is not None else el.get("content", "")
                    metadata = {"table_extractor": el["extracted"].get("extractor")} if el.get("extracted") is not None else {}
                elif el_type in FIGURE_TYPES:
                    last_anchor_type = "figure"
                    text = f"[FIGURE] {el.get('content', '')}"
                    metadata = {}
                elif el_type in FORMULA_TYPES:
                    last_anchor_type = "formula"
                    text = el.get("content", "") or ""
                    metadata = {}
                else:  # CAPTION_TYPES
                    text = el.get("content", "") or ""
                    metadata = {"caption_for": last_anchor_type} if last_anchor_type is not None else {}
                
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
                    metadata=metadata,
                ))
                seq += 1
                continue

            content = el.get("content", "") or ""
            current_embedding = embed_lookup.get(idx)
            current_tokens = count_tokens("\n\n".join(running_text_parts))

            should_flush_semantic = False
            should_flush_token = False

            if running_text_parts:
                if current_tokens >= self.min_chunk_tokens and last_embedding is not None and current_embedding is not None:
                    sim = cosine_similarity(last_embedding, current_embedding)
                    if sim < self.similarity_threshold:
                        should_flush_semantic = True
                
                prospective_tokens = count_tokens("\n\n".join(running_text_parts + [content]))
                if prospective_tokens > self.max_section_tokens:
                    should_flush_token = True

            if should_flush_token:
                self.token_flushes += 1
                flush()
            elif should_flush_semantic:
                self.semantic_flushes += 1
                flush()

            running_text_parts.append(content)
            running_indices.append(idx)
            running_bboxes.append(el["bbox"])
            last_embedding = current_embedding

        flush()
        return chunks
