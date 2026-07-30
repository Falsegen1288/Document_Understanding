from collections import defaultdict

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Standard RRF: for each ranked list, for each item at rank r (0-indexed),
    adds 1/(k + r + 1) to that item's cumulative score.
    Returns combined list sorted descending by score: [(chunk_id, score), ...]
    """
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked_list):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
