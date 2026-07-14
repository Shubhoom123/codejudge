"""Shared cleanup for raw LLM code output before it's treated as a solution.

Models routinely wrap output in markdown fences, and sometimes append
self-test/example code after the actual function definition — e.g. a
trailing `check_parse_amount()` call that references an unquoted `fifty`
instead of `"fifty"`. That crashes on import with a NameError, which
kills grading before a single real test case runs, and makes an
otherwise-correct solution look like a total failure.

Previously each caller (evaluator.py, tools/evaluate_at_scale.py,
tools/debug_model.py) had its own ad hoc stripping logic, and they'd
drifted apart — evaluator.py only stripped markdown fences and had no
protection against trailing self-test code at all. This module is the
single place that logic lives now.
"""
from __future__ import annotations

import ast


def strip_markdown_fences(raw: str) -> str:
    """Remove ``` / ```python fences wherever they appear, not just at the
    very start/end of the string (models don't always fence cleanly)."""
    raw = raw.strip()
    if "```" not in raw:
        return raw

    parts = raw.split("```")
    if len(parts) >= 3:
        code = parts[1]
        if code.startswith("python"):
            code = code[len("python"):].lstrip("\n")
        return code.strip()
    if len(parts) == 2:
        return parts[1].strip()
    return raw


def strip_top_level_side_effects(code: str) -> str:
    """Drop top-level statements that *execute* rather than *define*.

    Self-test calls, bare asserts, `print(...)`, `if __name__ == ...`
    blocks, and stray assignments are all top-level statements that can
    execute on import and crash — a defined-but-unused helper function
    can't. Uses the real Python parser (ast) rather than line-prefix
    heuristics, so it isn't fooled by unusual formatting.

    If the code doesn't parse as valid Python at all, it's returned
    unchanged — a syntax error will surface naturally once the grader
    actually tries to import it, which is a more honest failure mode
    than silently mangling unparseable input.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    keep_types = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    kept_nodes = [n for n in tree.body if isinstance(n, keep_types)]

    if len(kept_nodes) == len(tree.body):
        return code  # nothing was dropped

    return ast.unparse(ast.Module(body=kept_nodes, type_ignores=[]))


def clean_llm_code(raw: str) -> str:
    """Full cleanup pipeline: fences, then trailing side effects."""
    code = strip_markdown_fences(raw)
    code = strip_top_level_side_effects(code)
    return code.strip()
