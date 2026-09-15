"""
PAL (Program-Aided Language Model) Math Executor
Replaces the regex/symbolic arithmetic engine with LLM-generated, sandboxed
executable Python for financial arithmetic (YoY %, CAGR, deltas, ratios).

Drop-in target: src/arithmetic/symbolic_engine.py
"""
from __future__ import annotations

import ast
import math
import multiprocessing as mp
import re
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 1. Static safety check — reject anything with imports, attribute access to
#    dunders, exec/eval calls, or file/network primitives BEFORE execution.
# ---------------------------------------------------------------------------
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith,
    ast.Global, ast.Nonlocal, ast.Lambda,
    ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
    ast.Try, ast.Raise,
)
_FORBIDDEN_NAMES = {
    "__import__", "open", "exec", "eval", "compile", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "__builtins__", "os", "sys", "subprocess", "socket", "shutil",
}


class UnsafeCodeError(ValueError):
    pass


def static_safety_check(code: str) -> None:
    """Parse the code and reject anything outside a narrow arithmetic subset."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise UnsafeCodeError(f"Generated code does not parse: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeCodeError(f"Disallowed construct: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(f"Disallowed dunder attribute access: {node.attr}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(f"Disallowed identifier: {node.id}")
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _FORBIDDEN_NAMES:
                raise UnsafeCodeError(f"Disallowed call: {fn.id}")


# ---------------------------------------------------------------------------
# 2. Restricted execution namespace — math only, no builtins by default.
# ---------------------------------------------------------------------------
_SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "range": range, "float": float, "int": int,
    "pow": pow, "sorted": sorted,
}
_SAFE_GLOBALS = {"__builtins__": _SAFE_BUILTINS, "math": math}


def _worker(code: str, conn) -> None:
    """Runs in a separate process; writes the value bound to `result` back."""
    local_ns: dict[str, Any] = {}
    try:
        exec(compile(code, "<pal>", "exec"), dict(_SAFE_GLOBALS), local_ns)
        if "result" not in local_ns:
            conn.send(("error", "Generated code did not assign a `result` variable."))
            return
        conn.send(("ok", local_ns["result"]))
    except Exception as e:  # noqa: BLE001 — deliberately broad, sandboxed
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


@dataclass
class PALResult:
    success: bool
    value: Optional[float]
    raw_code: str
    error: Optional[str] = None


def run_pal_code(code: str, timeout_s: float = 3.0) -> PALResult:
    """
    Execute LLM-generated arithmetic code in an isolated subprocess with a
    hard wall-clock timeout. The code MUST assign its final answer to a
    variable named `result`.
    """
    try:
        static_safety_check(code)
    except UnsafeCodeError as e:
        return PALResult(success=False, value=None, raw_code=code, error=str(e))

    parent_conn, child_conn = mp.Pipe()
    proc = mp.Process(target=_worker, args=(code, child_conn))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return PALResult(success=False, value=None, raw_code=code,
                          error=f"Execution exceeded {timeout_s}s timeout")

    if not parent_conn.poll():
        return PALResult(success=False, value=None, raw_code=code,
                          error="Worker produced no output (crashed silently)")

    status, payload = parent_conn.recv()
    if status == "error":
        return PALResult(success=False, value=None, raw_code=code, error=payload)

    try:
        value = float(payload)
    except (TypeError, ValueError):
        return PALResult(success=False, value=None, raw_code=code,
                          error=f"`result` was not numeric: {payload!r}")
    return PALResult(success=True, value=value, raw_code=code)


# ---------------------------------------------------------------------------
# 3. Reader-facing entry point: prompt template + parse + execute.
#    Call this from benchmark_harness/stages/reading.py in place of the
#    existing regex-based SYMBOLIC arithmetic step.
# ---------------------------------------------------------------------------
PAL_PROMPT_TEMPLATE = """You are a financial calculation assistant. Given the
question and the retrieved context, write short Python code that computes the
numeric answer. Use only arithmetic and the `math` module. Assign the final
number to a variable named `result`. Do not import anything. Do not print.

Question: {question}

Context:
{context}

Python code:
```python
"""


def extract_code_block(llm_output: str) -> str:
    """Pull the first ```python ... ``` fenced block out of the LLM response."""
    match = re.search(r"```(?:python)?\s*(.*?)```", llm_output, re.DOTALL)
    return match.group(1).strip() if match else llm_output.strip()


def compute_pal_answer(question: str, context: str, llm_call_fn) -> PALResult:
    """
    llm_call_fn: Callable[[str], str] — your existing reader-LLM call
    (Qwen-2.5-32B / Groq client), injected so this module has no hard
    dependency on the specific provider.
    """
    prompt = PAL_PROMPT_TEMPLATE.format(question=question, context=context)
    raw = llm_call_fn(prompt)
    code = extract_code_block(raw)
    return run_pal_code(code)
