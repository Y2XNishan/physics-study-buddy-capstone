from __future__ import annotations

import ast
import operator
import re
from datetime import datetime


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculate_expression(question: str) -> str:
    try:
        # Strip common action keywords first
        cleaned = re.sub(r"(?i)^(calculate|solve|compute|what is|find)\s+", "", question.strip())
        cleaned = cleaned.rstrip("?.!")
        match = re.search(r"([-+/*()%\d.\s^]+)", cleaned)
        if not match:
            return "Calculator tool could not find a valid arithmetic expression."
        expression = match.group(1).strip().replace("^", "**")
        if not expression or not any(char.isdigit() for char in expression):
            return "Calculator tool could not find a valid arithmetic expression."
        value = _safe_eval(ast.parse(expression, mode="eval").body)
        return f"Calculator result: {expression.replace('**', '^')} = {value:.4f}"
    except Exception as exc:
        return f"Calculator tool error: {exc}"


def current_datetime_tool() -> str:
    now = datetime.now()
    return now.strftime("Current date and time: %A, %d %B %Y, %I:%M %p")


def choose_tool(question: str) -> str:
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

