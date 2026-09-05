from __future__ import annotations

import ast
import operator
import re
from datetime import datetime


import math

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "radians": math.radians,
    "degrees": math.degrees,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in SAFE_CONSTANTS:
        return float(SAFE_CONSTANTS[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
        args = [_safe_eval(arg) for arg in node.args]
        return float(SAFE_FUNCTIONS[node.func.id](*args))
    raise ValueError("Unsupported expression")


def calculate_expression(question: str) -> str:
    """Safely extract and evaluate a mathematical expression from the question string."""
    try:
        # Strip common action keywords first
        cleaned = re.sub(r"(?i)^(calculate|solve|compute|what is|find)\s+", "", question.strip())
        cleaned = cleaned.rstrip("?.!")
        match = re.search(r"([a-zA-Z0-9_+\-*/()%\s.^]+)", cleaned)
        if not match:
            return "Calculator tool could not find a valid arithmetic expression."
        expression = match.group(1).strip().replace("^", "**")
        if not expression or not any(char.isdigit() for char in expression):
            return "Calculator tool could not find a valid arithmetic expression."
        value = _safe_eval(ast.parse(expression, mode="eval").body)
        return f"Calculator result: {expression.replace('**', '^')} = {value:.4f}"
    except ZeroDivisionError:
        return "Calculator tool error: Division by zero is undefined."
    except OverflowError:
        return "Calculator tool error: Numerical result out of range (overflow)."
    except ValueError as exc:
        return f"Calculator tool error: {exc}"
    except Exception as exc:
        return f"Calculator tool error: {exc}"


def current_datetime_tool() -> str:
    """Return current system date and time formatted nicely."""
    now = datetime.now()
    return now.strftime("Current date and time: %A, %d %B %Y, %I:%M %p")


def choose_tool(question: str) -> str:
    """Disambiguate query intent and dispatch to date/time or calculator tool."""
    lowered = question.lower()
    if re.search(r"\b(date|today|clock)\b", lowered):
        return current_datetime_tool()
    if re.search(r"\b(time|day)\b", lowered):
        # Prevent physics terms containing 'time' from routing to datetime tool
        physics_time_terms = [
            "time period",
            "relaxation time",
            "decay time",
            "travel time",
            "time of flight",
            "position-time",
            "velocity-time",
            "time taken",
            "half life",
            "coherence time",
            "transit time",
        ]
        if not any(term in lowered for term in physics_time_terms):
            return current_datetime_tool()
    if any(token in lowered for token in ["calculate", "solve", "compute"]):
        return calculate_expression(question)
    if re.search(r"\d+\s*[-+/*^%]\s*\d+", lowered):
        return calculate_expression(question)
    return (
        "Tool route selected, but no supported tool matched the question. "
        "Available tools are calculator and current date/time."
    )

