"""Token counting utilities shared across all chunkers."""
import logging

logger = logging.getLogger(__name__)

_ENCODING = None
_TIKTOKEN_AVAILABLE = True

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken not installed — falling back to whitespace-split token "
        "approximation. Install tiktoken for accurate counts: pip install tiktoken"
    )


def count_tokens(text: str) -> int:
    """Return token count for `text`. Uses cl100k_base if tiktoken is available,
    otherwise a whitespace-split approximation (undercounts vs. real BPE tokenizers,
    treat as a rough heuristic only in degraded mode)."""
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE:
        return len(_ENCODING.encode(text))
    return len(text.split())


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` to at most `max_tokens` tokens. Used by strategies that need
    hard token ceilings (e.g. splitting oversized sections)."""
    if not _TIKTOKEN_AVAILABLE:
        words = text.split()
        return " ".join(words[:max_tokens])
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODING.decode(tokens[:max_tokens])
