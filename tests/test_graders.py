"""Regression tests for the graders themselves.

The graders (tasks/*/grader.py) are the ground truth for whether a model's
code is "correct." If a grader has a bug — a wrong expected value, an
overly strict/loose check — every score downstream is wrong too, silently.

Each test here runs a known-correct, hand-written reference solution
through the real grader and asserts a perfect score. If someone edits a
grader.py and breaks it, these fail immediately instead of quietly
corrupting a benchmark run.
"""
import pytest

import config
from evaluator import run_solution_against_grader

REFERENCE_SOLUTIONS = {
    "ambigous_spec": '''
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
''',
    "edge_case_trap": '''
from collections import Counter as _Counter

def most_frequent(lst):
    return _Counter(lst).most_common(1)[0][0]
''',
    "race_condition": '''
import threading

class Counter:
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._count += 1

    @property
    def value(self):
        return self._count
''',
}


@pytest.mark.parametrize("task", config.list_tasks())
def test_reference_solution_passes_its_grader(task):
    solution = REFERENCE_SOLUTIONS.get(task)
    if solution is None:
        pytest.skip(f"No reference solution registered for task '{task}' yet")

    scores = run_solution_against_grader(solution, config.load_task_grader(task))
    assert scores.get("score") == 1.0, scores


def test_every_task_has_a_reference_solution():
    """Guards against silently adding a task with no grader-correctness check."""
    missing = [t for t in config.list_tasks() if t not in REFERENCE_SOLUTIONS]
    assert not missing, f"Add a reference solution for: {missing}"
