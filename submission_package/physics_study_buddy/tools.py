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
    "sind": lambda deg: math.sin(math.radians(deg)),
    "cosd": lambda deg: math.cos(math.radians(deg)),
    "tand": lambda deg: math.tan(math.radians(deg)),
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "radians": math.radians,
    "degrees": math.degrees,
}

SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "g": 9.8,
    "c": 3e8,
    "h": 6.626e-34,
    "kB": 1.38e-23,
    "G": 6.674e-11,
    "e_charge": 1.602e-19,
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
        match = re.search(r"([a-zA-Z0-9_+\-*/()%\s.^eE]+)", cleaned)
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


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert common physical quantities between SI and non-SI units."""
    from_u, to_u = from_unit.strip().lower(), to_unit.strip().lower()
    if from_u == to_u:
        return f"Unit conversion result: {value} {from_unit} = {value:.4f} {to_unit}"

    # Base factors relative to SI standard units (m, kg, s, J)
    length = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "miles": 1609.34, "ft": 0.3048}
    mass = {"kg": 1.0, "g": 0.001, "mg": 1e-6, "lb": 0.453592}
    time_units = {"s": 1.0, "min": 60.0, "hr": 3600.0}
    energy = {"j": 1.0, "ev": 1.60218e-19, "cal": 4.184}

    for unit_dict in [length, mass, time_units, energy]:
        if from_u in unit_dict and to_u in unit_dict:
            val_in_si = value * unit_dict[from_u]
            result = val_in_si / unit_dict[to_u]
            return f"Unit conversion result: {value} {from_unit} = {result:.4e}" if abs(result) < 1e-3 or abs(result) > 1e4 else f"Unit conversion result: {value} {from_unit} = {result:.4f} {to_unit}"

    # Temperature explicit conversions
    if from_u in ["c", "celsius"] and to_u in ["k", "kelvin"]:
        return f"Unit conversion result: {value} °C = {value + 273.15:.2f} K"
    if from_u in ["k", "kelvin"] and to_u in ["c", "celsius"]:
        return f"Unit conversion result: {value} K = {value - 273.15:.2f} °C"

def vector_magnitude(vec: list[float]) -> float:
    """Calculate Euclidean norm/magnitude of a 2D or 3D vector."""
    return math.sqrt(sum(x * x for x in vec))


def vector_dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate dot product of two vectors of equal dimension."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimension for dot product.")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def vector_cross_product_2d(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate scalar cross product magnitude for 2D vectors (ax*by - ay*bx)."""
    if len(vec_a) < 2 or len(vec_b) < 2:
        raise ValueError("Vectors must have at least 2 components for cross product.")
    return vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]




def choose_tool(question: str) -> str:
    """Disambiguate query intent and dispatch to date/time, unit conversion, or calculator tool."""
    lowered = question.lower()
    conv_match = re.search(r"convert\s+([\d.]+)\s*([a-zA-Z°]+)\s+(?:to|in)\s+([a-zA-Z°]+)", lowered)
    if conv_match:
        try:
            val = float(conv_match.group(1))
            return convert_units(val, conv_match.group(2), conv_match.group(3))
        except ValueError:
            pass

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
        "Available tools are calculator, unit converter, and current date/time."
    )


