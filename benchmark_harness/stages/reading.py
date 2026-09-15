import os
import re
from typing import List, Dict, Any, Optional
from src.readers.window_reader import FastUpgradedWindowReader
from src.readers.llm_reader import LLMReader
from src.arithmetic.symbolic_engine import SymbolicArithmeticEngine


class ReadingStage:
    """
    Reading Stage: Uses Gemini 3.6 Flash as the primary reader model
    to extract facts, perform math, and synthesize answers from cross-linked chunks.
    """

    def __init__(
        self,
        use_llm_reader: bool = True,
        use_pal_arithmetic: bool = False,
        reader_model: Optional[str] = "gemini-3.6-flash",
    ):
        self.window_reader = FastUpgradedWindowReader()
        self.arithmetic_engine = SymbolicArithmeticEngine()
        self.reader_model = reader_model or "gemini-3.6-flash"
        self.llm_reader = None

        if use_llm_reader:
            try:
                self.llm_reader = LLMReader(model=self.reader_model)
            except Exception as e:
                print(f"[READING WARNING] Could not initialize LLMReader ({e}).", flush=True)

    def extract_answer(self, query: str, context_chunks: List[str], query_type: str = "prose") -> Dict[str, Any]:
        full_context = "\n\n".join(context_chunks)

        # 1. Primary Path: Generative LLM Reader (Gemini 3.6 Flash)
        llm_ans = None
        if self.llm_reader is not None:
            llm_res = self.llm_reader.answer(query, context_chunks, query_type=query_type)
            llm_ans = llm_res.get("answer")

        # 2. Extractive & Symbolic Fallbacks (Active only if LLM is offline or returns NOT_FOUND)
        is_multi_span = "multi-span" in str(query_type or "").lower()
        extractive_ans = self.window_reader.extract_answer_span(query, full_context, is_multi_span=is_multi_span)

        symbolic_ans = None
        q_type_str = str(query_type or "").lower()
        if "arithmetic" in q_type_str or any(op in query.lower() for op in ["change", "diff", "sum", "total", "ratio"]):
            raw_nums = re.findall(r'\b[\$\u20ac\u00a3]?\(?\d[\d,.]*\)?%?\b', full_context)
            cleaned_nums = [self.arithmetic_engine.clean_financial_number(n) for n in raw_nums]
            valid_nums = [n for n in cleaned_nums if n is not None]
            if valid_nums:
                symbolic_ans = str(valid_nums[0])

        # Select Final Primary Answer
        if llm_ans and not llm_ans.startswith("NOT_FOUND"):
            primary = llm_ans
        elif symbolic_ans and "arithmetic" in q_type_str:
            primary = symbolic_ans
        else:
            primary = extractive_ans

        return {
            "extractive_answer": extractive_ans,
            "symbolic_answer": symbolic_ans,
            "llm_answer": llm_ans,
            "primary_answer": primary
        }
