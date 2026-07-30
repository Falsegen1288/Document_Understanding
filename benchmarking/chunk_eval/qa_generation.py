"""Generates a QA eval set directly from Stage 1 elements, before any chunking is
applied. Each question is tagged with the exact source_element_indices that contain
the answer."""
import os
import sys
# Add project root to path for imports
sys.path.append(os.path.abspath('c:/Users/user/Downloads/Document_Understanding'))

import json
import logging
from pathlib import Path
from groq import Groq
from algorithms.config import GROQ_API_KEY
from chunking.element_types import TABLE_TYPES, FIGURE_TYPES, TEXT_LIKE_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTION_TYPES = ["factual_text", "table_lookup", "figure_description"]

GENERATION_PROMPT_TEMPLATE = """You are generating a retrieval-evaluation question from a
single document element. Given the element type and content below, write ONE specific
question that can ONLY be answered using this exact content (not general knowledge), plus
a concise reference answer.

Element type: {element_type}
Element content:
{content}

Respond in strict JSON, no markdown fences, no preamble:
{{"question": "...", "reference_answer": "..."}}
"""

def select_source_elements(stage1_json: dict, n_per_type: int = 5) -> list[dict]:
    """Pick a spread of elements to generate questions from: some plain text, some
    tables, some figures."""
    elements = stage1_json["elements"]
    by_type = {"factual_text": [], "table_lookup": [], "figure_description": []}
    for idx, el in enumerate(elements):
        el_type = el["type"]
        
        # Check content length appropriately
        if el_type in TABLE_TYPES:
            content = el["extracted"]["markdown"] if "extracted" in el else el.get("content", "")
        else:
            content = el.get("content", "") or ""
            
        if not content or len(content) < 20:
            continue
            
        if el_type in TEXT_LIKE_TYPES:
            by_type["factual_text"].append((idx, el))
        elif el_type in TABLE_TYPES:
            by_type["table_lookup"].append((idx, el))
        elif el_type in FIGURE_TYPES:
            by_type["figure_description"].append((idx, el))

    selected = []
    for qtype, items in by_type.items():
        # Pick first n_per_type items
        for idx, el in items[:n_per_type]:
            selected.append({"question_type": qtype, "element_idx": idx, "element": el})
    return selected


def generate_qa_for_document(doc_stem: str, llm_call_fn, n_per_type: int = 5) -> list[dict]:
    with open(f"outputs/{doc_stem}/{doc_stem}.json", encoding="utf-8") as f:
        stage1 = json.load(f)

    candidates = select_source_elements(stage1, n_per_type=n_per_type)
    qa_pairs = []
    for cand in candidates:
        el = cand["element"]
        content = el["extracted"]["markdown"] if el["type"] in TABLE_TYPES and "extracted" in el else el.get("content", "")
        prompt = GENERATION_PROMPT_TEMPLATE.format(element_type=el["type"], content=content)
        try:
            raw = llm_call_fn(prompt)
            # Clean JSON markdown blocks if any
            cleaned = raw.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.removeprefix("```json")
            if cleaned.endswith("```"):
                cleaned = cleaned.removesuffix("```")
            cleaned = cleaned.strip()
            
            parsed = json.loads(cleaned)
            qa_pairs.append({
                "doc_stem": doc_stem,
                "question_type": cand["question_type"],
                "question": parsed["question"],
                "reference_answer": parsed["reference_answer"],
                "source_element_indices": [cand["element_idx"]],
                "source_page": el["page"],
            })
            logger.info(f"Generated QA for {doc_stem} element {cand['element_idx']} ({cand['question_type']})")
        except Exception as e:
            logger.warning(f"QA generation failed for {doc_stem} element {cand['element_idx']}: {e}. Raw response: {raw if 'raw' in locals() else 'None'}")
    return qa_pairs


def run_generation():
    client = Groq(api_key=GROQ_API_KEY)
    
    def llm_call(prompt: str) -> str:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    docs = ["Medical_004_demo_30p", "Researchpaper_KAI", "Scientific_001"]
    all_qa = []
    # Generate 6 questions per type per document
    for doc in docs:
        logger.info(f"Starting QA generation for {doc}...")
        qa_pairs = generate_qa_for_document(doc, llm_call, n_per_type=6)
        all_qa.extend(qa_pairs)

    out_path = Path("benchmarking/chunk_eval/qa_eval_set.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
        
    print(f"Generated a total of {len(all_qa)} questions.")
    
    # Print breakdown
    breakdown = {}
    for q in all_qa:
        breakdown[q["question_type"]] = breakdown.get(q["question_type"], 0) + 1
    print("Breakdown:", breakdown)

if __name__ == "__main__":
    run_generation()
