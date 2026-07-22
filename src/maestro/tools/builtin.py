"""A small, safe set of built-in tools.

Deliberately conservative — no shell, no arbitrary file writes, no network by
default — so a swarm can be handed tools without opening a hole.  Richer/riskier
capabilities are better delegated to a sibling agent (I.S.A.A.C. has a sandbox).
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator as _op

from maestro.tools.base import FunctionTool, Tool

# -- calculator (safe arithmetic via AST, no eval) ---------------------------
_BINOPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow,
}
_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _safe_eval(expr: str) -> str:
    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return str(_ev(tree.body))


def _now(_arg: str) -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _wordcount(text: str) -> str:
    return str(len(text.split()))


_BUILTINS: dict[str, Tool] = {
    "calc": FunctionTool(
        "calc", "Evaluate a simple arithmetic expression, e.g. '2*(3+4)'.", _safe_eval
    ),
    "now": FunctionTool("now", "Return the current local date-time (ISO 8601).", _now),
    "wordcount": FunctionTool(
        "wordcount", "Count the words in the argument text.", _wordcount
    ),
}


def builtin_tools() -> dict[str, Tool]:
    return dict(_BUILTINS)
