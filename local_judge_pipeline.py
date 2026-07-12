"""
local_judge_pipeline.py

Drop-in replacement for the Groq-hosted, rate-limited judge stack in
run_chunking_eval.py. Removes the TokenRateLimiter entirely by pointing
Ragas + DeepEval at a locally-hosted, OpenAI-API-compatible vLLM server
instead of the Groq API.
"""

import asyncio
import json
import os
import hashlib
import sqlite3
import logging
from typing import Any, List, Optional

from openai import AsyncOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage
from deepeval.models import DeepEvalBaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Shared local client -- points at vLLM (or Ollama) instead of Groq.
# No API key, no TPM quota, no 429s.
# ---------------------------------------------------------------------
LOCAL_JUDGE_BASE_URL = os.environ.get("LOCAL_JUDGE_BASE_URL", "http://localhost:8000/v1")
LOCAL_JUDGE_MODEL = os.environ.get("LOCAL_JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
CACHE_DB = os.environ.get("LOCAL_JUDGE_CACHE_DB", "local_judge_cache.db")

_client = AsyncOpenAI(base_url=LOCAL_JUDGE_BASE_URL, api_key="not-needed")

# Cap in-flight requests to something sane for the GPU/VRAM rather than an
# artificial token quota -- vLLM's scheduler handles the actual batching.
_MAX_CONCURRENT_REQUESTS = int(os.environ.get("LOCAL_JUDGE_CONCURRENCY", "64"))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


# ---------------------------------------------------------------------
# Persistent Cache Database Setup
# ---------------------------------------------------------------------
def init_cache_db():
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS judge_cache (
            prompt_hash TEXT PRIMARY KEY,
            prompt TEXT,
            response TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize cache at import time
init_cache_db()

def get_cached_response(prompt: str) -> Optional[str]:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    try:
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM judge_cache WHERE prompt_hash = ?", (prompt_hash,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        logger.warning(f"Failed to query SQLite cache: {e}")
        return None

def set_cached_response(prompt: str, response: str):
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    try:
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO judge_cache (prompt_hash, prompt, response) VALUES (?, ?, ?)", (prompt_hash, prompt, response))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to write to SQLite cache: {e}")


def delete_cached_response(prompt: str):
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    try:
        conn = sqlite3.connect(CACHE_DB)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM judge_cache WHERE prompt_hash = ?", (prompt_hash,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to delete from SQLite cache: {e}")


async def _call_local_judge(prompt: str, model_name: str = LOCAL_JUDGE_MODEL, json_mode: bool = True) -> str:
    # Prioritize environment variable if explicitly set, otherwise use model_name
    actual_model = os.environ.get("LOCAL_JUDGE_MODEL", model_name)

    # Check cache first to bypass LLM call if possible
    cached = get_cached_response(prompt)
    if cached is not None:
        return cached

    async with _semaphore:
        response = await _client.chat.completions.create(
            model=actual_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"} if json_mode else None,
        )
        result = response.choices[0].message.content
        # Store in cache
        set_cached_response(prompt, result)
        return result


# ---------------------------------------------------------------------
# Ragas wrapper -- replaces CustomGroqChatModel
# ---------------------------------------------------------------------
class LocalJudgeChatModel(BaseChatModel):
    """LangChain-compatible chat model backed by the local vLLM server.
    No rate limiter needed: concurrency is bounded by _semaphore above,
    and the vLLM server itself batches everything it receives."""

    model_name: str = LOCAL_JUDGE_MODEL

    @property
    def _llm_type(self) -> str:
        return "local-vllm-judge"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        prompt = "\n".join(m.content for m in messages)
        content = asyncio.run(_call_local_judge(prompt, model_name=self.model_name))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        prompt = "\n".join(m.content for m in messages)
        content = await _call_local_judge(prompt, model_name=self.model_name)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


# ---------------------------------------------------------------------
# DeepEval wrapper -- replaces CustomDeepEvalLLM
# ---------------------------------------------------------------------
class LocalJudgeDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self, model_name: str = LOCAL_JUDGE_MODEL):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def generate(self, prompt: str) -> str:
        return asyncio.run(_call_local_judge(prompt, model_name=self.model_name))

    async def a_generate(self, prompt: str) -> str:
        return await _call_local_judge(prompt, model_name=self.model_name)

    def get_model_name(self) -> str:
        return self.model_name


# ---------------------------------------------------------------------
# Consolidated Call Optimization helper functions
# ---------------------------------------------------------------------
CONSOLIDATED_JUDGE_PROMPT = """You are an expert RAG evaluation judge. Given a question, \
a generated response, the ground truth answer, and the retrieved context chunks, score \
ALL of the following metrics on a 0.0-1.0 scale. Apply the same standard both frameworks \
would use; scores should be consistent between the "ragas_" and "deepeval_" keys unless \
the response genuinely trades off differently against ground truth vs. context.

QUESTION:
{question}

CONTEXT CHUNKS:
{context}

GENERATED RESPONSE:
{response}

GROUND TRUTH:
{ground_truth}

Return ONLY a JSON object with exactly this shape, no other text:
{{
  "ragas_faithfulness": 0.0,
  "ragas_answer_relevancy": 0.0,
  "ragas_context_precision": 0.0,
  "ragas_context_recall": 0.0,
  "deepeval_faithfulness": 0.0,
  "deepeval_answer_relevancy": 0.0,
  "deepeval_contextual_precision": 0.0,
  "deepeval_contextual_recall": 0.0
}}"""

REQUIRED_METRIC_KEYS = [
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_context_precision",
    "ragas_context_recall",
    "deepeval_faithfulness",
    "deepeval_answer_relevancy",
    "deepeval_contextual_precision",
    "deepeval_contextual_recall"
]

async def score_question_consolidated(
    question: str, context: str, response: str, ground_truth: str
) -> dict:
    prompt = CONSOLIDATED_JUDGE_PROMPT.format(
        question=question, context=context, response=response, ground_truth=ground_truth
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw = await _call_local_judge(prompt, json_mode=True)
            data = json.loads(raw)
            
            # Validate all required keys are present, numeric, and bounded
            valid = True
            for k in REQUIRED_METRIC_KEYS:
                if k not in data:
                    logger.warning(f"Missing key '{k}' in consolidated judge output on attempt {attempt + 1}")
                    valid = False
                    break
                val = data[k]
                if not isinstance(val, (int, float)):
                    logger.warning(f"Key '{k}' value '{val}' is not numeric on attempt {attempt + 1}")
                    valid = False
                    break
                if not (0.0 <= float(val) <= 1.0):
                    logger.warning(f"Key '{k}' value '{val}' is out of bounds [0.0, 1.0] on attempt {attempt + 1}")
                    valid = False
                    break
            
            if valid:
                return {k: float(data[k]) for k in REQUIRED_METRIC_KEYS}
                
            # If invalid, remove from cache to ensure we do not store bad completions
            delete_cached_response(prompt)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed to score consolidated: {e}")
            delete_cached_response(prompt)
            
        await asyncio.sleep(1.0)
        
    logger.error("All consolidated judge scoring attempts failed. Returning fallback 0.0 scores.")
    return {k: 0.0 for k in REQUIRED_METRIC_KEYS}


async def run_consolidated_eval(golden_qa_items: List[dict]) -> List[dict]:
    """
    golden_qa_items: list of dicts, each:
        {"question": ..., "context": ..., "response": ..., "ground_truth": ...}
    (the already-retrieved/generated data from pipeline steps 1-4, before
    handing off to Ragas/DeepEval).

    Fires all 54 questions concurrently; vLLM's continuous batching packs
    them onto the GPU. Replaces 432 throttled sequential calls with 54
    unthrottled concurrent calls.
    """
    tasks = [
        score_question_consolidated(
            item["question"], item["context"], item["response"], item["ground_truth"]
        )
        for item in golden_qa_items
    ]
    return await asyncio.gather(*tasks)
