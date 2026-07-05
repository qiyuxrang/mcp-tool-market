"""
Calculator MCP Server

Provides two tools:
- calculate: Safely evaluate mathematical expressions via AST (not eval()).
- unit_convert: Convert between common units of length, weight, and temperature.
"""

import ast
import math
import operator
import os
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Safe AST-based expression evaluator
# ---------------------------------------------------------------------------

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node):
    """Recursively evaluate an AST node, allowing only safe operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARY:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return ALLOWED_UNARY[op_type](operand)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        try:
            return ALLOWED_OPERATORS[op_type](left, right)
        except ZeroDivisionError:
            raise ValueError("Division by zero")

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def safe_eval(expr: str):
    """Parse and safely evaluate a mathematical expression string using AST."""
    tree = ast.parse(expr.strip(), mode="eval")
    try:
        return _eval_node(tree.body)
    except RecursionError:
        raise ValueError("Expression too complex or deeply nested")


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def _format_number(n):
    """Format a number with up to 6 decimal places, stripping trailing zeros."""
    if isinstance(n, float):
        if not math.isfinite(n):
            return str(n)  # inf, -inf, nan pass through as-is
        if n == 0.0:
            return "0"
        if abs(n) < 1e-8 or abs(n) > 1e15:
            formatted = f"{n:.6e}"
            mantissa, exp = formatted.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            return f"{mantissa}e{exp}"
        formatted = f"{n:.6f}".rstrip("0").rstrip(".")
        return formatted
    return str(n)


# ---------------------------------------------------------------------------
# Unit definitions
# ---------------------------------------------------------------------------

# Length: canonical factor relative to 1 meter
LENGTH = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}

# Weight: canonical factor relative to 1 gram
WEIGHT = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
}

# Temperature: maps aliases to canonical names for conversion logic
TEMPERATURE = {
    "c": "celsius",
    "celsius": "celsius",
    "f": "fahrenheit",
    "fahrenheit": "fahrenheit",
    "k": "kelvin",
    "kelvin": "kelvin",
}

UNIT_CATEGORIES = {
    "length": LENGTH,
    "weight": WEIGHT,
    "temperature": TEMPERATURE,
}

# Reverse lookup: alias  ->  category name
ALIAS_TO_CATEGORY: dict[str, str] = {}
for category, units in UNIT_CATEGORIES.items():
    for alias in units:
        ALIAS_TO_CATEGORY[alias.lower()] = category


def _find_category(unit: str) -> str | None:
    """Return the category name for a unit alias, or None if not recognised."""
    return ALIAS_TO_CATEGORY.get(unit.lower())


def _convert_length(value: float, from_unit: str, to_unit: str) -> float:
    from_factor = LENGTH[from_unit.lower()]
    to_factor = LENGTH[to_unit.lower()]
    return value * from_factor / to_factor


def _convert_weight(value: float, from_unit: str, to_unit: str) -> float:
    from_factor = WEIGHT[from_unit.lower()]
    to_factor = WEIGHT[to_unit.lower()]
    return value * from_factor / to_factor


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    from_canon = TEMPERATURE[from_unit.lower()]
    to_canon = TEMPERATURE[to_unit.lower()]

    # Convert to Celsius first
    if from_canon == "celsius":
        celsius = value
    elif from_canon == "fahrenheit":
        celsius = (value - 32) * 5.0 / 9.0
    else:  # kelvin
        celsius = value - 273.15

    # Convert from Celsius to target
    if to_canon == "celsius":
        return celsius
    if to_canon == "fahrenheit":
        return celsius * 9.0 / 5.0 + 32
    # kelvin
    return celsius + 273.15


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("Calculator Server")


@mcp.tool()
def calculate(expr: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports operators: +, -, *, /, **, %, //, parentheses, and unary minus/plus.
    Uses Python's AST module for safe parsing — no eval() is used.
    """
    original = expr.strip()
    if not original:
        return "Error: Empty expression"

    try:
        result = safe_eval(original)
        formatted = _format_number(result)
        return f"{original} = {formatted}"
    except (ValueError, SyntaxError) as e:
        return f"Error: {e}"


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between supported units.

    Supported categories and units:
    - Length: m (meter), km (kilometer), cm (centimeter), in (inch), ft (foot), mi (mile)
    - Weight: g (gram), kg (kilogram), lb (pound)
    - Temperature: c (celsius), f (fahrenheit), k (kelvin)

    Common plural/singular aliases are recognised for each unit.
    """
    from_norm = from_unit.strip().lower()
    to_norm = to_unit.strip().lower()

    from_cat = _find_category(from_norm)
    to_cat = _find_category(to_norm)

    if from_cat is None:
        return f"Error: Unknown unit '{from_unit}'"
    if to_cat is None:
        return f"Error: Unknown unit '{to_unit}'"
    if from_cat != to_cat:
        return f"Error: Cannot convert between '{from_unit}' ({from_cat}) and '{to_unit}' ({to_cat})"

    try:
        if from_cat == "length":
            result = _convert_length(value, from_norm, to_norm)
        elif from_cat == "weight":
            result = _convert_weight(value, from_norm, to_norm)
        else:  # temperature
            result = _convert_temperature(value, from_norm, to_norm)

        formatted_value = _format_number(value)
        formatted_result = _format_number(result)
        return f"{formatted_value} {from_unit} = {formatted_result} {to_unit}"
    except (ValueError,) as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8003))
    import uvicorn
    uvicorn.run(mcp.sse_app(), host="0.0.0.0", port=port)
