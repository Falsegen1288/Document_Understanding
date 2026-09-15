"""
Scorer Reweighting
Fixes: UniDoc-Bench headline Token F1 (~0.035) understating true quality
because gold answers are 25-45 word explanatory sentences while the reader
extracts short factual spans. Containment/Precision (~18-24%) show the
extraction is actually correct.

Drop-in target: benchmark_harness/stages/evaluation.py / benchmark_harness/stages/scorer_reweight.py
This does NOT remove Token F1/Recall/ROUGE-L from the report -- they stay
as secondary diagnostics. It adds a `headline_score` that stakeholders
should read as the primary quality number.
"""
from __future__ import annotations

from dataclasses import dataclass


# Weights sum to 1.0. LLM Judge is our primary quality arbiter (evaluates factual
# adequacy, completeness, and reasoning), followed by Containment and Numeric EM.
HEADLINE_WEIGHTS = {
    "llm_judge": 0.40,
    "containment": 0.25,
    "numeric_em": 0.25,
    "token_f1": 0.10,
}

DIAGNOSTIC_ONLY = {"token_f1", "token_precision", "token_recall", "rouge_l", "set_em", "char_similarity"}


@dataclass
class ScoreBundle:
    llm_judge: float
    set_em: float
    numeric_em: float
    token_f1: float
    token_precision: float
    token_recall: float
    rouge_l: float
    char_similarity: float
    containment: float
    semantic_score: float

    def headline_score(self, weights: dict[str, float] = HEADLINE_WEIGHTS) -> float:
        return (
            weights.get("llm_judge", 0.40) * self.llm_judge
            + weights.get("containment", 0.25) * self.containment
            + weights.get("numeric_em", 0.25) * self.numeric_em
            + weights.get("token_f1", 0.10) * self.token_f1
        )

    def as_report_dict(self) -> dict:
        """
        Returns metrics grouped by role, so downstream reporting/dashboards
        can visually separate "trust this" from "diagnostic only" without
        deleting any existing metric.
        """
        return {
            "headline_score": round(self.headline_score(), 4),
            "primary": {
                "llm_judge": self.llm_judge,
                "containment": self.containment,
                "numeric_em": self.numeric_em,
            },
            "diagnostic_only": {
                "set_em": self.set_em,
                "token_f1": self.token_f1,
                "token_precision": self.token_precision,
                "token_recall": self.token_recall,
                "rouge_l": self.rouge_l,
                "char_similarity": self.char_similarity,
                "semantic_score": self.semantic_score,
            },
        }


# Resolves key aliases and fails loudly on missing keys.
_KEY_ALIASES: dict[str, list[str]] = {
    "llm_judge": ["llm_judge", "llm_judge_score", "semantic_score"],
    "set_em": ["set_em", "set_exact_match"],
    "numeric_em": ["numeric_em", "numeric_exact_match"],
    "token_f1": ["token_f1", "f1"],
    "token_precision": ["token_precision", "precision"],
    "token_recall": ["token_recall", "recall"],
    "rouge_l": ["rouge_l"],
    "char_similarity": ["char_similarity", "character_similarity"],
    "containment": ["containment", "answer_containment"],
    "semantic_score": ["semantic_score", "llm_judge"],
}


def _resolve(run_metrics: dict, field: str) -> float:
    for key in _KEY_ALIASES[field]:
        if key in run_metrics:
            return float(run_metrics[key])
    raise KeyError(
        f"None of the expected keys {_KEY_ALIASES[field]} found for field "
        f"'{field}'. Keys actually present: {list(run_metrics.keys())}"
    )


def recompute_headline_from_existing_run(run_metrics: dict) -> dict:
    """
    Utility to re-derive headline_score from an ALREADY-COMPLETED benchmark
    run's saved JSON without re-running the harness.
    """
    bundle = ScoreBundle(
        llm_judge=_resolve(run_metrics, "llm_judge"),
        set_em=_resolve(run_metrics, "set_em"),
        numeric_em=_resolve(run_metrics, "numeric_em"),
        token_f1=_resolve(run_metrics, "token_f1"),
        token_precision=_resolve(run_metrics, "token_precision"),
        token_recall=_resolve(run_metrics, "token_recall"),
        rouge_l=_resolve(run_metrics, "rouge_l"),
        char_similarity=_resolve(run_metrics, "char_similarity"),
        containment=_resolve(run_metrics, "containment"),
        semantic_score=_resolve(run_metrics, "semantic_score"),
    )
    return bundle.as_report_dict()
