import os
import sys
import re
import math
import ast
from typing import List, Dict, Any, Tuple, Union

class SymbolicArithmeticEngine:
    """
    Production Symbolic Arithmetic Execution Engine for Financial Table & Document RAG.
    Executes symbolic computations (SUM, DIFFERENCE, AVERAGE, RATIO, PERCENTAGE_CHANGE, MULTI_STEP)
    over extracted table cell values with scale & financial unit normalization.
    """
    
    # Financial Scale Multipliers
    SCALE_MAP = {
        "thousand": 1000.0,
        "thousands": 1000.0,
        "million": 1000000.0,
        "millions": 1000000.0,
        "billion": 1000000000.0,
        "billions": 1000000000.0,
        "percent": 0.01,
        "%": 0.01
    }

    def __init__(self, numeric_tolerance: float = 0.01):
        self.numeric_tolerance = numeric_tolerance

    @classmethod
    def clean_financial_number(cls, raw_val: Union[str, int, float]) -> Union[float, None]:
        """
        Parses financial number strings into floats.
        Handles parenthetical negatives: '(2,088)' -> -2088.0
        Handles commas and currency symbols: '$1,945,391' -> 1945391.0
        """
        if raw_val is None:
            return None
        if isinstance(raw_val, (int, float)):
            return float(raw_val)

        val_str = str(raw_val).strip()
        if not val_str:
            return None

        # Parenthetical negative check
        is_negative = False
        paren_match = re.search(r'\(([\d,.]+)\)', val_str)
        if paren_match:
            is_negative = True
            val_str = paren_match.group(1)
        elif val_str.startswith('-'):
            is_negative = True
            val_str = val_str[1:]

        # Strip currency symbols, commas, spaces, %
        clean_str = re.sub(r'[\$,\s%]', '', val_str)
        try:
            num = float(clean_str)
            return -num if is_negative else num
        except ValueError:
            return None

    @classmethod
    def extract_numbers_from_text(cls, text: str) -> List[float]:
        """Extracts all clean numerical values from text or table row/column representations."""
        # Find integers or floating point numbers (including parenthetical negative numbers)
        tokens = re.findall(r'\(?\b\d[\d,.]*\b\)?', text)
        nums = []
        for t in tokens:
            parsed = cls.clean_financial_number(t)
            if parsed is not None:
                nums.append(parsed)
        return nums

    @classmethod
    def classify_operation(cls, derivation_str: str) -> str:
        """Classifies TAT-DQA derivation strings into operation taxonomy."""
        d = derivation_str.strip() if derivation_str else ""
        if not d:
            return "UNKNOWN"
        
        has_plus = '+' in d
        has_minus = '-' in d
        has_slash = '/' in d
        
        if (has_plus or has_minus) and has_slash and ('(' in d or '[' in d):
            if re.search(r'/\s*\d+\b', d):
                return "AVERAGE"
            if '-' in d and '/' in d:
                return "PERCENTAGE_CHANGE"
            return "MULTI_STEP"
        
        if has_plus and not has_minus and not has_slash:
            return "SUM"
        if has_minus and not has_plus and not has_slash:
            return "DIFFERENCE"
        if has_slash and not has_plus and not has_minus:
            return "RATIO"
        if has_plus:
            return "SUM"
        if has_minus:
            return "DIFFERENCE"
        if has_slash:
            return "RATIO"
        return "OTHER"

    @classmethod
    def safe_eval_expression(cls, expr: str) -> Union[float, None]:
        """Safely evaluates numeric arithmetic expressions via AST."""
        try:
            # Clean expression string
            clean_expr = re.sub(r'[\[\{]', '(', expr)
            clean_expr = re.sub(r'[\]\}]', ')', clean_expr)
            # Replace numbers with commas e.g. 14,740 -> 14740
            clean_expr = re.sub(r'(\d+),(\d+)', r'\1\2', clean_expr)
            
            node = ast.parse(clean_expr, mode='eval')
            
            def eval_node(n):
                if isinstance(n, ast.Expression):
                    return eval_node(n.body)
                elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    return float(n.value)
                elif isinstance(n, ast.UnaryOp):
                    operand = eval_node(n.operand)
                    if isinstance(n.op, ast.USub):
                        return -operand
                    elif isinstance(n.op, ast.UAdd):
                        return operand
                elif isinstance(n, ast.BinOp):
                    left = eval_node(n.left)
                    right = eval_node(n.right)
                    if isinstance(n.op, ast.Add):
                        return left + right
                    elif isinstance(n.op, ast.Sub):
                        return left - right
                    elif isinstance(n.op, ast.Mult):
                        return left * right
                    elif isinstance(n.op, ast.Div):
                        return left / right if right != 0 else None
                return None

            return eval_node(node)
        except Exception:
            return None

    def execute_derivation(self, numbers: List[float], op_type: str) -> Union[float, None]:
        """
        Executes symbolic computation over numeric operands.
        """
        if not numbers:
            return None

        if op_type == "SUM":
            return sum(numbers)
        elif op_type == "DIFFERENCE":
            if len(numbers) >= 2:
                return numbers[0] - numbers[1]
            return numbers[0]
        elif op_type == "AVERAGE":
            return sum(numbers) / float(len(numbers))
        elif op_type == "RATIO":
            if len(numbers) >= 2 and numbers[1] != 0:
                return numbers[0] / numbers[1]
            return None
        elif op_type == "PERCENTAGE_CHANGE":
            if len(numbers) >= 2 and numbers[0] != 0:
                # (New - Old) / Old
                return (numbers[1] - numbers[0]) / numbers[0]
            return None
        elif op_type == "MULTI_STEP":
            if len(numbers) >= 4:
                avg1 = (numbers[0] + numbers[1]) / 2.0
                avg2 = (numbers[2] + numbers[3]) / 2.0
                return avg1 - avg2
            elif len(numbers) >= 2:
                return numbers[0] - numbers[1]
        
        return sum(numbers)

    def check_numeric_match(self, pred: Union[float, int, str], gt: Union[float, int, str], scale: str = None) -> bool:
        """
        Checks numeric exact match under specified tolerance.
        Numeric Exact Match Tolerance: abs(pred - gt) <= 0.01 or relative error <= 0.001.
        """
        pred_num = self.clean_financial_number(pred)
        gt_num = self.clean_financial_number(gt)

        if pred_num is None or gt_num is None:
            return False

        # Scale adjustment if pred is in percent decimal e.g. 0.125 vs 12.5%
        diff = abs(pred_num - gt_num)
        if diff <= self.numeric_tolerance:
            return True

        # Relative error check
        rel_err = diff / max(1.0, abs(gt_num))
        if rel_err <= 0.001:
            return True

        # Check percentage representation e.g. 0.125 vs 12.5
        if abs((pred_num * 100.0) - gt_num) <= self.numeric_tolerance or abs((gt_num * 100.0) - pred_num) <= self.numeric_tolerance:
            return True

        return False
