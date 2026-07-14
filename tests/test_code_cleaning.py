from code_cleaning import clean_llm_code, strip_markdown_fences, strip_top_level_side_effects


def test_strip_markdown_fences_python_tagged():
    raw = "```python\ndef f():\n    return 1\n```"
    assert strip_markdown_fences(raw) == "def f():\n    return 1"


def test_strip_markdown_fences_untagged():
    raw = "```\ndef f():\n    return 1\n```"
    assert strip_markdown_fences(raw) == "def f():\n    return 1"


def test_strip_markdown_fences_no_fences_is_noop():
    raw = "def f():\n    return 1"
    assert strip_markdown_fences(raw) == raw


def test_strip_top_level_side_effects_removes_bare_call():
    code = "def f():\n    return 1\n\nf()\n"
    cleaned = strip_top_level_side_effects(code)
    # the definition survives; the top-level call (a side effect) doesn't
    assert "def f" in cleaned
    assert cleaned.strip() == "def f():\n    return 1"


def test_strip_top_level_side_effects_removes_if_name_main_block():
    code = 'def f():\n    return 1\n\nif __name__ == "__main__":\n    print(f())\n'
    cleaned = strip_top_level_side_effects(code)
    assert "__main__" not in cleaned
    assert "def f" in cleaned


def test_strip_top_level_side_effects_keeps_valid_definitions_only_code_unchanged():
    code = "import re\n\ndef f(x):\n    return re.sub('a', 'b', x)\n"
    assert strip_top_level_side_effects(code) == code


def test_strip_top_level_side_effects_returns_unparseable_code_unchanged():
    broken = "def f(:\n    this is not python"
    assert strip_top_level_side_effects(broken) == broken


def test_regression_self_test_with_undefined_name_no_longer_crashes_on_import():
    """Regression test for a real failure: qwen2-codejudge generated a
    correct parse_amount, then appended a self-test that referenced a
    bare, unquoted `fifty` (instead of "fifty"). That crashed the module
    on import with a NameError before any real grader test ran — an
    otherwise-working solution got scored as a total failure because of
    unrelated trailing code."""
    raw = '''```python
def parse_amount(text: str) -> float:
    return float(text.replace("$", ""))

def check_parse_amount():
    assert parse_amount("50") == 50.0, "Test 1 failed"
    assert parse_amount("$50") == 50.0, "Test 2 failed"
    assert parse_amount(fifty) == 50.0, "Test 3 failed"

check_parse_amount()
```'''
    cleaned = clean_llm_code(raw)

    # Must be importable now — the crashing top-level call is gone.
    namespace: dict = {}
    exec(compile(cleaned, "<solution>", "exec"), namespace)
    assert namespace["parse_amount"]("$50") == 50.0


def test_clean_llm_code_full_pipeline():
    raw = "```python\ndef f():\n    return 1\n\nprint(f())\n```"
    cleaned = clean_llm_code(raw)
    assert cleaned == "def f():\n    return 1"
