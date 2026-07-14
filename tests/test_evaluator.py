import pytest

import config
from evaluator import load_task, resolve_model, run_solution_against_grader

# Same reference implementation used to generate/validate the bulk test
# suite in tools/generate_test_cases.py — known to pass all 24 grader
# cases for ambigous_spec.
REFERENCE_PARSE_AMOUNT = '''
def parse_amount(text: str) -> float:
    text = text.strip().lower()
    if not text:
        raise ValueError("Empty input")

    text = text.replace("$", "").replace(",", "")
    if "dollars" in text:
        text = text.replace("dollars", "").strip()
    if "dollar" in text:
        text = text.replace("dollar", "").strip()
    if "usd" in text:
        text = text.replace("usd", "").strip()

    text = text.strip()

    try:
        return float(text)
    except ValueError:
        pass

    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
        "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90,
    }

    words = text.replace(" and ", " ").split()
    total = 0
    current = 0
    for word in words:
        if word in number_words:
            current += number_words[word]
        elif word == "hundred":
            current = current * 100 if current else 100
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
        elif word == "million":
            total += (current or 1) * 1000000
            current = 0
        else:
            raise ValueError(f"Cannot parse: {word!r}")

    result = total + current
    if result == 0 and text not in ("zero", "0", "$0", "0.0"):
        raise ValueError(f"Cannot parse: {text!r}")
    return float(result)
'''


def test_load_task_reads_prompt_and_grader():
    task = load_task("ambigous_spec")
    assert "parse_amount" in task["prompt"]
    assert task["grader_path"].exists()


def test_load_task_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_task("does_not_exist")


def test_correct_solution_scores_perfectly():
    scores = run_solution_against_grader(
        REFERENCE_PARSE_AMOUNT, config.load_task_grader("ambigous_spec")
    )
    assert scores["score"] == 1.0, scores


def test_naive_solution_scores_low():
    """qwen2:1.5b's actual baseline failure mode — only handles plain floats."""
    naive = "def parse_amount(text: str) -> float:\n    return float(text)\n"
    scores = run_solution_against_grader(naive, config.load_task_grader("ambigous_spec"))
    assert scores["score"] < 0.5, scores


def test_non_python_output_is_reported_not_crashed():
    scores = run_solution_against_grader(
        "this is not python at all !!!", config.load_task_grader("ambigous_spec")
    )
    # Should come back as a structured error/failure, never raise out of
    # run_solution_against_grader itself.
    assert "error" in scores or scores.get("passed", 1) == 0


def test_infinite_loop_is_killed_not_hung():
    """An infinite loop should never make the harness itself hang.

    It may be stopped by the wall-clock timeout or by the sandbox's CPU
    time limit (whichever fires first) — either way this call must return
    within a bounded time and report a failure, not hang the test suite.
    """
    stuck = "def parse_amount(text):\n    while True:\n        pass\n"
    scores = run_solution_against_grader(stuck, config.load_task_grader("ambigous_spec"))
    assert "error" in scores


def test_resolve_model_known_key():
    name, provider = resolve_model("gemma3:12b")
    assert provider == "ollama"
