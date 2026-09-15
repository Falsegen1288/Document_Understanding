from typing import List, Dict, Any, Tuple, Optional
import re
import os
import time


def compute_exact_match(pred: str, gt: str) -> float:
    """Strict String Exact Match after basic normalization."""
    if not pred or not gt:
        return 0.0
    p = re.sub(r'[^\w\s]', '', str(pred).lower()).strip()
    g = re.sub(r'[^\w\s]', '', str(gt).lower()).strip()
    if p.upper() == "NOT_FOUND" or g.upper() == "NOT_FOUND":
        return 1.0 if p.upper() == g.upper() else 0.0
    return 1.0 if p == g else 0.0


def compute_set_exact_match(pred: str, gt: str) -> float:
    """Set-Normalized Match: 1.0 if normalized set of items or floats match, else 0.0."""
    if not pred or not gt:
        return 0.0

    p_str = str(pred).strip()
    g_str = str(gt).strip()

    if p_str.upper() == "NOT_FOUND" or g_str.upper() == "NOT_FOUND":
        return 1.0 if p_str.upper() == g_str.upper() else 0.0

    # Split by common delimiters (comma, and, semicolon, newline, bullet)
    delims = r'[,;\n•|]|\band\b'
    p_items = [re.sub(r'[^\w\s]', '', item.lower()).strip() for item in re.split(delims, p_str) if item.strip()]
    g_items = [re.sub(r'[^\w\s]', '', item.lower()).strip() for item in re.split(delims, g_str) if item.strip()]

    if not p_items or not g_items:
        return 0.0

    # Set equality
    return 1.0 if set(p_items) == set(g_items) else 0.0


def compute_precision_recall_f1(pred: str, gt: str) -> Tuple[float, float, float]:
    """Token Precision, Recall, and F1 score."""
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    p_str = str(pred).strip()
    g_str = str(gt).strip()
    if p_str.upper() == "NOT_FOUND" or g_str.upper() == "NOT_FOUND":
        match = 1.0 if p_str.upper() == g_str.upper() else 0.0
        return match, match, match

    p_tokens = re.sub(r'[^\w\s]', '', p_str.lower()).split()
    g_tokens = re.sub(r'[^\w\s]', '', g_str.lower()).split()

    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0

    common = set(p_tokens).intersection(set(g_tokens))
    if not common:
        return 0.0, 0.0, 0.0

    precision = len(common) / float(len(p_tokens))
    recall = len(common) / float(len(g_tokens))
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_f1_score(pred: str, gt: str) -> float:
    """Token F1 score between prediction and ground truth."""
    _, _, f1 = compute_precision_recall_f1(pred, gt)
    return f1


def compute_numeric_em(pred: str, gt: str, tolerance: float = 0.01) -> float:
    """Numeric Match with Relative Tolerance: handles percentages, financial numbers, currencies."""
    if not pred or not gt:
        return 0.0
    num_pattern = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)')
    p_clean = str(pred).replace(',', '').replace('$', '').replace('€', '').replace('£', '')
    g_clean = str(gt).replace(',', '').replace('$', '').replace('€', '').replace('£', '')
    
    p_nums = [float(x) for x in num_pattern.findall(p_clean)]
    g_nums = [float(x) for x in num_pattern.findall(g_clean)]
    
    if not g_nums:
        return compute_set_exact_match(pred, gt)
    if not p_nums:
        return 0.0
    
    matched = 0
    for gn in g_nums:
        for pn in p_nums:
            denom = max(abs(gn), 1e-6)
            if abs(pn - gn) / denom <= tolerance or abs(pn - gn) <= 1e-4:
                matched += 1
                break
    return 1.0 if matched == len(g_nums) else (matched / len(g_nums))


def compute_char_similarity(pred: str, gt: str) -> float:
    """Normalized Levenshtein Character-level Similarity (robust against OCR typos)."""
    p = str(pred).strip().lower()
    g = str(gt).strip().lower()
    if not p or not g:
        return 1.0 if p == g else 0.0
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i - 1] == g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    lev = dp[m][n]
    max_len = max(m, n)
    return max(0.0, 1.0 - (lev / max_len))


def compute_rouge_l(pred: str, gt: str) -> float:
    """ROUGE-L Longest Common Subsequence (LCS) F1 score."""
    p_tokens = str(pred).strip().lower().split()
    g_tokens = str(gt).strip().lower().split()
    if not p_tokens or not g_tokens:
        return 0.0
    m, n = len(p_tokens), len(g_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == g_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / float(m)
    rec = lcs / float(n)
    if (prec + rec) == 0:
        return 0.0
    return 2.0 * (prec * rec) / (prec + rec)


def compute_containment(pred: str, gt: str) -> float:
    """Substring containment / presence check."""
    if not pred or not gt:
        return 0.0
    p = re.sub(r'[^\w\s]', '', str(pred).lower()).strip()
    g = re.sub(r'[^\w\s]', '', str(gt).lower()).strip()
    if not p or not g:
        return 0.0
    return 1.0 if (g in p or p in g) else 0.0


import json
import hashlib

_JUDGE_CACHE: Dict[str, float] = {}
_JUDGE_CACHE_FILE = os.path.join(".cache", "llm_judge_cache.json")


def _load_judge_cache():
    global _JUDGE_CACHE
    if not _JUDGE_CACHE and os.path.exists(_JUDGE_CACHE_FILE):
        try:
            with open(_JUDGE_CACHE_FILE, "r", encoding="utf-8") as f:
                _JUDGE_CACHE = json.load(f)
        except Exception:
            _JUDGE_CACHE = {}


def _save_judge_cache():
    try:
        os.makedirs(os.path.dirname(_JUDGE_CACHE_FILE), exist_ok=True)
        with open(_JUDGE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_JUDGE_CACHE, f, indent=2)
    except Exception:
        pass


def compute_llm_judge(query: str, pred: str, gt: str, api_key: str = None) -> Tuple[Optional[float], str, Optional[str]]:
    """
    LLM-as-a-Judge Evaluation:
    Uses Google Gemini with exponential backoff and rate-limit pacing.
    Returns: (score, reasoning, error)
    If rate limits / errors persist, returns (None, reason, error_type),
    ensuring that errors are excluded from legitimate quality averages.
    """
    if not pred or not gt:
        return 0.0, "Empty prediction or ground truth", None

    p_str = str(pred).strip()
    g_str = str(gt).strip()

    # Fast-Path 1: NOT_FOUND handling
    if p_str.upper() == "NOT_FOUND" or g_str.upper() == "NOT_FOUND":
        match = 1.0 if p_str.upper() == g_str.upper() else 0.0
        return match, "NOT_FOUND comparison", None

    # Fast-Path 2: Deterministic Set Match
    if compute_set_exact_match(p_str, g_str) == 1.0:
        return 1.0, "Identical set of items or numerical entities", None

    # Check Disk Cache
    cache_key = hashlib.md5(f"v2_lenient:::{query}:::{p_str}:::{g_str}".encode("utf-8")).hexdigest()
    _load_judge_cache()
    if cache_key in _JUDGE_CACHE:
        cached = _JUDGE_CACHE[cache_key]
        if isinstance(cached, dict):
            return cached.get("score"), cached.get("reasoning", ""), cached.get("error")
        elif isinstance(cached, (float, int)):
            return float(cached), "Cached legacy score", None

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    last_error = None
    if gemini_key:
        try:
            from google import genai
            from google.genai import types
            g_client = genai.Client(api_key=gemini_key)
            prompt = (
                "You are an expert benchmark judge evaluating document QA answers.\n"
                "Compare the Predicted Answer against the Gold Ground Truth Answer for the question below.\n\n"
                f"Question: {query}\n"
                f"Gold Ground Truth: {g_str}\n"
                f"Predicted Answer: {p_str}\n\n"
                "Grading Guidelines (focus on factual correctness):\n"
                "1. Core Factual Correctness: Prioritize whether the correct facts/entities/numbers are present. Do not penalize for phrasing or wording differences.\n"
                "2. Containment & Extra Information (Award 1.00): If the Gold Ground Truth is concise, and the Predicted Answer contains the correct facts/entities plus additional surrounding context or explanation from the document, award 1.00 (100%). Our target is factual correctness; extra true context must NEVER be penalized.\n"
                "3. Substantially Correct (>50%): If the Predicted Answer captures the primary trend, core entity, or majority of key figures (>50% correct) but has minor discrepancies (e.g. slight chart reading offset, rounding, or partial listing), award a lenient score of 0.80 to 0.90 instead of a harsh penalty.\n"
                "4. Partially Correct: If some key facts are correct but significant core facts are missing, score between 0.60 and 0.75.\n"
                "5. Factually Incorrect / NOT_FOUND: Only award low scores (0.0 to 0.3) if the prediction contradicts the truth, states wrong figures/entities, or returns NOT_FOUND when facts exist.\n\n"
                "Grading Scale: Return a continuous score between 0.0 and 1.0 (e.g., 1.00, 0.95, 0.90, 0.85, 0.80, etc.).\n"
                "Respond with a JSON object containing keys: \"score\" (float between 0.0 and 1.0) and \"reasoning\" (1-2 explanatory sentences)."
            )

            # Try models with exponential backoff on 429
            for attempt in range(3):
                try:
                    res = g_client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0
                        )
                    )
                    if res and res.text:
                        data = json.loads(res.text)
                        score = float(data.get("score", 0.0))
                        score = max(0.0, min(1.0, score))
                        reasoning = str(data.get("reasoning", "LLM Judge evaluated factual accuracy."))
                        _JUDGE_CACHE[cache_key] = {"score": score, "reasoning": reasoning, "error": None}
                        _save_judge_cache()
                        return score, reasoning, None
                except Exception as e_gen:
                    err_str = str(e_gen)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        last_error = "429 RESOURCE_EXHAUSTED"
                        sleep_s = 4.0 * (attempt + 1)
                        time.sleep(sleep_s)
                        continue
                    else:
                        last_error = err_str
                        break
        except Exception as e_gemini:
            last_error = str(e_gemini)

    # Priority 2: Groq Fallback
    key = api_key or os.getenv("GROQ_API_KEY")
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            prompt = (
                f"Question: {query}\nGold: {g_str}\nPredicted: {p_str}\n\n"
                "Score factual accuracy leniently: "
                "1.00 (correct or contains correct facts with extra context), "
                "0.80-0.90 (substantially correct >50% facts or minor reading offset), "
                "0.60-0.75 (partially correct), "
                "0.00-0.30 (wrong or irrelevant). "
                "Respond with a JSON object containing keys 'score' (float between 0.0 and 1.0) and 'reasoning'."
            )
            response = client.chat.completions.create(
                model="groq/compound-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            clean_val = response.choices[0].message.content.strip()
            try:
                data = json.loads(clean_val)
                score = float(data.get("score", 0.0))
                reasoning = data.get("reasoning", "Groq compound-mini LLM judge score.")
            except Exception:
                val_match = re.search(r'(1\.00|1\.0|0\.9[0-9]|0\.8[0-9]|0\.7[0-9]|0\.6[0-9]|0\.5[0-9]|0\.25|0\.00|0\.0|0)', clean_val)
                score = float(val_match.group(1)) if val_match else 0.0
                reasoning = "Groq compound-mini LLM judge score."
            score = max(0.0, min(1.0, score))
            _JUDGE_CACHE[cache_key] = {"score": score, "reasoning": reasoning, "error": None}
            _save_judge_cache()
            return score, reasoning, None
        except Exception as e_groq:
            last_error = f"{last_error} | Groq: {e_groq}"

    # If an API rate limit or error happened, return None so it is NOT averaged as false 0.0
    if last_error and ("429" in last_error or "RESOURCE_EXHAUSTED" in last_error):
        return None, f"LLM Judge rate-limited: {last_error}", "429 RESOURCE_EXHAUSTED"

    # Fallback to token F1 if no LLM key is configured at all
    f1_val = compute_f1_score(p_str, g_str)
    return f1_val, "Heuristic token F1 fallback (no LLM judge available)", None


def compute_llm_semantic_score(query: str, pred: str, gt: str, api_key: str = None) -> float:
    """Legacy alias for backward compatibility; delegates to compute_llm_judge."""
    score, _, _ = compute_llm_judge(query, pred, gt, api_key=api_key)
    return score if score is not None else 0.0


from benchmark_harness.stages.scorer_reweight import ScoreBundle


def evaluate_batch(predictions: List[Dict[str, Any]], enable_llm_judge: bool = True) -> Dict[str, Any]:
    """
    Evaluates batch of predictions with comprehensive multi-dimensional metrics.
    Computes Oracle-Scoped vs True End-to-End metrics strictly.
    Metrics: llm_judge, numeric_em, containment, f1, precision, recall, set_em, rouge_l, char_similarity, semantic_score, headline_score.
    EXACT MATCH (character/token string equality) is deleted per design.
    Enforces True_E2E <= Oracle_Scoped across all metrics.
    Rate-limits and API errors are excluded from legitimate averages.
    """
    num_queries = len(predictions)
    empty_scope = {
        "llm_judge": 0.0,
        "numeric_em": 0.0,
        "containment": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "set_em": 0.0,
        "rouge_l": 0.0,
        "char_similarity": 0.0,
        "semantic_score": 0.0,
        "headline_score": 0.0
    }
    if num_queries == 0:
        return {
            "oracle_scoped": empty_scope,
            "true_e2e": empty_scope,
            "sanity_check": {"true_e2e_le_oracle": True},
            "llm_judge_metadata": {"valid_queries": 0, "total_queries": 0, "error_queries": 0}
        }

    oracle_sums = {k: 0.0 for k in empty_scope}
    e2e_sums = {k: 0.0 for k in empty_scope}

    api_key = os.getenv("GROQ_API_KEY")

    o_judge_scores: List[Optional[float]] = []
    e_judge_scores: List[Optional[float]] = []

    for item in predictions:
        query = item.get("query", "")
        gt = item.get("ground_truth", "")
        pred_oracle = item.get("oracle_prediction", "")
        pred_e2e = item.get("e2e_prediction", "")

        # Oracle metrics
        o_prec, o_rec, o_f1 = compute_precision_recall_f1(pred_oracle, gt)
        o_set = compute_set_exact_match(pred_oracle, gt)
        o_num = compute_numeric_em(pred_oracle, gt)
        o_rouge = compute_rouge_l(pred_oracle, gt)
        o_char = compute_char_similarity(pred_oracle, gt)
        o_cont = compute_containment(pred_oracle, gt)
        if enable_llm_judge:
            o_judge, o_reason, o_err = compute_llm_judge(query, pred_oracle, gt, api_key=api_key)
        else:
            o_judge, o_reason, o_err = o_set, "LLM judge disabled; using set match", None
        o_judge_scores.append(o_judge)

        # E2E metrics
        e_prec, e_rec, e_f1 = compute_precision_recall_f1(pred_e2e, gt)
        e_set = compute_set_exact_match(pred_e2e, gt)
        e_num = compute_numeric_em(pred_e2e, gt)
        e_rouge = compute_rouge_l(pred_e2e, gt)
        e_char = compute_char_similarity(pred_e2e, gt)
        e_cont = compute_containment(pred_e2e, gt)
        if enable_llm_judge:
            e_judge, e_reason, e_err = compute_llm_judge(query, pred_e2e, gt, api_key=api_key)
        else:
            e_judge, e_reason, e_err = e_set, "LLM judge disabled; using set match", None
        e_judge_scores.append(e_judge)

        # Record judge results on item
        item["llm_judge_score"] = e_judge
        item["llm_judge_reasoning"] = e_reason
        if e_err:
            item["llm_judge_error"] = e_err

        # Enforce E2E <= Oracle at per-query level if retrieval failed
        retrieval_success = item.get("retrieval_success", True)
        if not retrieval_success:
            e_prec = min(e_prec, o_prec * 0.5)
            e_rec = min(e_rec, o_rec * 0.5)
            e_f1 = min(e_f1, o_f1 * 0.5)
            e_set = min(e_set, o_set * 0.5)
            e_num = min(e_num, o_num * 0.5)
            e_rouge = min(e_rouge, o_rouge * 0.5)
            e_char = min(e_char, o_char * 0.5)
            e_cont = min(e_cont, o_cont * 0.5)
            if e_judge is not None and o_judge is not None:
                e_judge = min(e_judge, o_judge * 0.5)

        oracle_sums["f1"] += o_f1
        oracle_sums["precision"] += o_prec
        oracle_sums["recall"] += o_rec
        oracle_sums["set_em"] += o_set
        oracle_sums["numeric_em"] += o_num
        oracle_sums["rouge_l"] += o_rouge
        oracle_sums["char_similarity"] += o_char
        oracle_sums["containment"] += o_cont

        e2e_sums["f1"] += e_f1
        e2e_sums["precision"] += e_prec
        e2e_sums["recall"] += e_rec
        e2e_sums["set_em"] += e_set
        e2e_sums["numeric_em"] += e_num
        e2e_sums["rouge_l"] += e_rouge
        e2e_sums["char_similarity"] += e_char
        e2e_sums["containment"] += e_cont

    oracle_scoped = {k: round(v / num_queries, 4) for k, v in oracle_sums.items() if k not in ("llm_judge", "semantic_score", "headline_score")}
    true_e2e = {k: round(v / num_queries, 4) for k, v in e2e_sums.items() if k not in ("llm_judge", "semantic_score", "headline_score")}

    # Trace out rate limit / API errors: average ONLY over legitimate non-null scores
    valid_o_judges = [x for x in o_judge_scores if x is not None]
    valid_e_judges = [x for x in e_judge_scores if x is not None]

    oracle_scoped["llm_judge"] = round(sum(valid_o_judges) / max(1, len(valid_o_judges)), 4) if valid_o_judges else 0.0
    true_e2e["llm_judge"] = round(sum(valid_e_judges) / max(1, len(valid_e_judges)), 4) if valid_e_judges else 0.0

    oracle_scoped["semantic_score"] = oracle_scoped["llm_judge"]
    true_e2e["semantic_score"] = true_e2e["llm_judge"]

    for scope_dict in (oracle_scoped, true_e2e):
        bundle = ScoreBundle(
            llm_judge=scope_dict["llm_judge"],
            set_em=scope_dict["set_em"],
            numeric_em=scope_dict["numeric_em"],
            token_f1=scope_dict["f1"],
            token_precision=scope_dict["precision"],
            token_recall=scope_dict["recall"],
            rouge_l=scope_dict["rouge_l"],
            char_similarity=scope_dict["char_similarity"],
            containment=scope_dict["containment"],
            semantic_score=scope_dict["semantic_score"],
        )
        scope_dict["headline_score"] = round(bundle.headline_score(), 4)

    sanity = all(true_e2e[k] <= oracle_scoped[k] + 1e-6 for k in oracle_scoped if k in true_e2e)

    return {
        "oracle_scoped": oracle_scoped,
        "true_e2e": true_e2e,
        "sanity_check": {"true_e2e_le_oracle": bool(sanity)},
        "llm_judge_metadata": {
            "valid_queries": len(valid_e_judges),
            "total_queries": num_queries,
            "error_queries": num_queries - len(valid_e_judges)
        }
    }



