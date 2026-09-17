"""A small, safe set of built-in tools.

Deliberately conservative — no shell, no arbitrary file writes, no network by
default — so a swarm can be handed tools without opening a hole.  Richer/riskier
capabilities are better delegated to a sibling agent (I.S.A.A.C. has a sandbox).
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator as _op

from maestro.tools.base import FunctionTool, Tool, ToolError

# -- calculator (safe arithmetic via AST, no eval) ---------------------------
_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}

_MAX_NODES = 200
_MAX_EXPR_LEN = 1000
_MAX_DEPTH = 64
_MAX_POW_EXP = 100
_MAX_INT_BITS = 4096


def _safe_eval(expr: str) -> str:
    """Evaluate simple arithmetic without ``eval``.

    Raises :class:`ToolError` (surfaced as ``"error: …"`` by
    :class:`FunctionTool`) on syntax errors, unsupported syntax,
    division-by-zero, runaway recursion, oversized expressions, or
    denial-of-service shapes like ``2**2**10``.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ToolError("empty expression")
    if len(expr) > _MAX_EXPR_LEN:
        raise ToolError(f"expression too long (max {_MAX_EXPR_LEN} chars)")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ToolError(f"invalid expression: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolError(f"invalid expression: {exc}") from exc

    try:
        node_count = sum(1 for _ in ast.walk(tree))
    except RecursionError as exc:
        raise ToolError("expression too deeply nested") from exc
    if node_count > _MAX_NODES:
        raise ToolError(f"expression too complex (max {_MAX_NODES} nodes)")

    def _check_size(value: int | float) -> None:
        if isinstance(value, bool):
            raise ToolError("unsupported expression")
        if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
            raise ToolError("result too large")
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ToolError("result is not finite")

    def _ev(node, depth: int = 0):
        if depth > _MAX_DEPTH:
            raise ToolError("expression too deeply nested")
        if isinstance(node, ast.Expression):
            return _ev(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ToolError("unsupported expression")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left = _ev(node.left, depth + 1)
            right = _ev(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if isinstance(right, float) and (right != right or abs(right) == float("inf")):
                    raise ToolError("invalid exponent")
                try:
                    exp_abs = abs(right)
                except Exception as exc:
                    raise ToolError(f"invalid exponent: {exc}") from exc
                if exp_abs > _MAX_POW_EXP:
                    raise ToolError(
                        f"exponent too large (max {_MAX_POW_EXP}); split the computation"
                    )
            try:
                result = _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError as exc:
                raise ToolError("division by zero") from exc
            except (OverflowError, ValueError, ArithmeticError) as exc:
                raise ToolError(f"arithmetic error: {exc}") from exc
            _check_size(result)
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            try:
                result = _UNARY[type(node.op)](_ev(node.operand, depth + 1))
            except (OverflowError, ValueError, ArithmeticError) as exc:
                raise ToolError(f"arithmetic error: {exc}") from exc
            _check_size(result)
            return result
        raise ToolError("unsupported expression")

    try:
        return str(_ev(tree.body))
    except ToolError:
        raise
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except RecursionError as exc:
        raise ToolError("expression too deeply nested") from exc
    except (OverflowError, ValueError, ArithmeticError) as exc:
        raise ToolError(f"arithmetic error: {exc}") from exc


def _now(_arg: str) -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _wordcount(text: str) -> str:
    return str(len(text.split()))


_BUILTINS: dict[str, Tool] = {
    "calc": FunctionTool(
        "calc", "Evaluate a simple arithmetic expression, e.g. '2*(3+4)'.", _safe_eval
    ),
    "now": FunctionTool("now", "Return the current local date-time (ISO 8601).", _now),
    "wordcount": FunctionTool("wordcount", "Count the words in the argument text.", _wordcount),
}


def builtin_tools() -> dict[str, Tool]:
    return dict(_BUILTINS)
