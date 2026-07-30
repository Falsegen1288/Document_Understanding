"""Common interface all Stage 2 chunking strategies must implement."""
from abc import ABC, abstractmethod

from chunking.schema import Chunk


class BaseChunker(ABC):
    """All chunkers consume a full Stage 1 result.json (already parsed into a dict)
    and return a flat list of Chunk objects across all pages of that document.
    Implementations must NOT mutate the input dict."""

    name: str  # must be set by subclasses, used as Chunk.strategy and in filenames

    @abstractmethod
    def chunk(self, stage1_json: dict) -> list[Chunk]:
        """Produce chunks for one document.

        Args:
            stage1_json: parsed Stage 1 result.json, matching the schema in the
                Stage 2 project spec (`metadata`, `elements`, `tables`,
                `visual_captions` keys).

        Returns:
            List of Chunk objects, ordered by (page, reading position). Every
            element in stage1_json["elements"] must be accounted for in exactly
            one chunk's source_element_indices — no dropped elements, no
            duplicated elements across chunks (validate this in Phase 2's tests).
        """
        raise NotImplementedError

    def _make_chunk_id(self, doc_stem: str, page: int, seq: int) -> str:
        """Shared ID formatter so all strategies produce consistent, sortable IDs."""
        return f"{doc_stem}_{self.name}_{page}_{seq:04d}"
