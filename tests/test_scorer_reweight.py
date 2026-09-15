import pytest
from benchmark_harness.stages.scorer_reweight import recompute_headline_from_existing_run, ScoreBundle

def test_recompute_headline_run1():
    run1_e2e = {
        "exact_match": 0.0015,
        "set_exact_match": 0.0015,
        "numeric_exact_match": 0.0353,
        "token_f1": 0.0351,
        "token_precision": 0.1686,
        "token_recall": 0.0347,
        "rouge_l": 0.0289,
        "character_similarity": 0.0573,
        "answer_containment": 0.1890,
        "semantic_score": 0.0351,
    }
    result = recompute_headline_from_existing_run(run1_e2e)
    assert result["headline_score"] == 0.089

def test_recompute_with_live_evaluation_keys():
    live_e2e = {
        "exact_match": 0.0015,
        "set_em": 0.0015,
        "numeric_em": 0.0353,
        "f1": 0.0351,
        "precision": 0.1686,
        "recall": 0.0347,
        "rouge_l": 0.0289,
        "char_similarity": 0.0573,
        "containment": 0.1890,
        "semantic_score": 0.0351,
    }
    result = recompute_headline_from_existing_run(live_e2e)
    assert result["headline_score"] == 0.089

def test_missing_key_raises_keyerror():
    broken = {
        "exact_match": 0.0,
        # missing set_em, numeric_em, etc.
    }
    with pytest.raises(KeyError) as exc_info:
        recompute_headline_from_existing_run(broken)
    assert "set_em" in str(exc_info.value)
