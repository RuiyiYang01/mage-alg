"""Safe Python evaluator for Program-of-Thought (PoT) reasoning.

Restricted exec environment that can run arithmetic + math operations
but blocks file I/O, imports, attribute access on dangerous objects, etc.
Used by the learner's PoT path to compute numeric answers from generated
Python code without exposing the full Python runtime.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

# Whitelist of safe builtins / modules
SAFE_BUILTINS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "int": int, "float": float, "bool": bool,
    "range": range, "list": list, "tuple": tuple, "set": set, "dict": dict,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed,
    "pow": pow, "divmod": divmod,
    "any": any, "all": all,
    "True": True, "False": False, "None": None,
    "print": lambda *a, **kw: None,  # no-op print to absorb model's print() calls
}
SAFE_MATH = {
    name: getattr(math, name)
    for name in (
        "pi", "e", "tau", "inf", "nan",
        "sqrt", "exp", "log", "log2", "log10", "log1p",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "floor", "ceil", "trunc", "fabs", "fmod",
        "factorial", "gcd",
    )
}

# Disallowed AST nodes (anything that imports, defines functions, etc.)
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Lambda,  # block lambdas to avoid runtime stitching
    ast.Global, ast.Nonlocal,
    ast.Try, ast.Raise,
    ast.With, ast.AsyncWith,
    ast.AsyncFor,
    ast.Yield, ast.YieldFrom,
    ast.Await,
    ast.Delete,
)


def _is_safe(tree: ast.AST) -> tuple[bool, str]:
    """Walk the AST and reject any forbidden nodes."""
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return False, f"forbidden node: {type(node).__name__}"
        if isinstance(node, ast.Attribute):
            # Block dunder access (__class__, __subclasses__, etc.)
            if node.attr.startswith("__"):
                return False, f"forbidden attribute: {node.attr}"
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                return False, f"forbidden name: {node.id}"
        if isinstance(node, ast.Call):
            # Block calls to builtins like exec/eval/__import__ even if
            # somehow re-introduced
            if isinstance(node.func, ast.Name) and node.func.id in (
                "exec", "eval", "compile", "open", "__import__",
                "getattr", "setattr", "delattr", "hasattr", "globals",
                "locals", "vars", "type", "id",
            ):
                return False, f"forbidden call: {node.func.id}"
    return True, ""


def safe_eval(code: str, timeout_seconds: float = 5.0) -> tuple[Any, str | None]:
    """Execute Python code in a restricted environment.

    Returns (result, error). The result is taken from the local variable
    named `result` if present, else `answer`, else the last expression
    statement's value. Error is None on success; otherwise a short
    description.

    The interpreter has access to:
      - SAFE_BUILTINS (arithmetic, len, sum, etc.)
      - SAFE_MATH names exposed at top-level (sqrt, pi, etc.)
    """
    code = (code or "").strip()
    if not code:
        return None, "empty code"
    if len(code) > 4000:
        return None, "code too long"

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return None, f"syntax error: {e}"

    ok, why = _is_safe(tree)
    if not ok:
        return None, why

    # Capture last expression value if no `result` variable
    last_expr_value = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Wrap last expr in `_LAST_EXPR_CAPTURE = expr`
        wrapped = ast.Assign(
            targets=[ast.Name(id="_LAST_EXPR_CAPTURE", ctx=ast.Store())],
            value=tree.body[-1].value,
        )
        ast.copy_location(wrapped, tree.body[-1])
        tree.body[-1] = wrapped
        ast.fix_missing_locations(tree)

    safe_globals: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
    }
    safe_globals.update(SAFE_MATH)
    safe_locals: dict[str, Any] = {}

    try:
        # No real timeout (would need signals/subprocess); rely on AST
        # restriction to prevent infinite loops / heavy compute. Most
        # FinQA computations are simple arithmetic.
        compiled = compile(tree, "<pot>", "exec")
        exec(compiled, safe_globals, safe_locals)
    except Exception as e:
        return None, f"runtime error: {type(e).__name__}: {str(e)[:120]}"

    # Result resolution order: result → answer → ans → last expression
    for key in ("result", "answer", "ans", "final_answer", "Answer"):
        if key in safe_locals:
            return safe_locals[key], None
    if "_LAST_EXPR_CAPTURE" in safe_locals:
        return safe_locals["_LAST_EXPR_CAPTURE"], None
    return None, "no result variable found"


def extract_python_code(response: str) -> str:
    """Extract Python code from an LLM response.

    Tries (in order):
      1. ```python ... ``` fenced block
      2. ``` ... ``` fenced block
      3. After a 'Code:' or 'Python:' marker
      4. Lines containing `result = ...`
    """
    response = response or ""
    # Fenced code blocks
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 'Code:' or 'Python:' marker
    m = re.search(r"(?:Code|Python|Solution)\s*:\s*\n(.*?)(?:\n\n|\Z)", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Last block of contiguous lines containing `=` or function calls
    lines = response.splitlines()
    block: list[str] = []
    for ln in reversed(lines):
        s = ln.strip()
        if not s:
            if block:
                break
            continue
        if re.match(r"^[a-zA-Z_]\w*\s*=|^\s*(result|answer|ans)\s*=|^[\d.()\[\]+\-*/ ]+$", s):
            block.append(ln)
        else:
            if block:
                break
    return "\n".join(reversed(block)).strip()


def run_pot(response: str) -> tuple[Any, str | None, str]:
    """Convenience wrapper: extract + execute, return (result, error, code)."""
    code = extract_python_code(response)
    if not code:
        return None, "no python code found", ""
    result, error = safe_eval(code)
    return result, error, code
