"""
Uses gemma3:12b (teacher) to generate high quality training data
for the ambigous_spec task, specifically targeting the failure
categories of qwen2:1.5b:
  - dollar sign ($12.50)
  - comma thousands ($1,000)
  - whitespace (  $42  )
  - word dollars (100 dollars)
  - written numbers (fifty, one hundred)
  - negative amounts (-$50)

Output: data/training_runs/training_data.json
        data/training_runs/train.jsonl  (Colab/HuggingFace format)
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
from evaluator import run_solution_against_grader

OUTPUT_DIR = Path("data/training_runs")
OLLAMA_URL = "http://localhost:11434/api/generate"
TEACHER_MODEL = "gemma3:12b"
ATTEMPTS = 100

TASK_DIR = Path("tasks/ambigous_spec")

PROMPT_HINT = """
You are an expert Python developer. Write a function `parse_amount(text: str) -> float`
that extracts a monetary amount from a user-typed string.

The function MUST handle ALL of these formats correctly:
  "$12.50"          -> 12.5      (dollar sign)
  "$1,000"          -> 1000.0    (comma thousands)
  "100 dollars"     -> 100.0     (word dollars)
  "100 dollar"      -> 100.0     (singular dollar)
  "fifty"           -> 50.0      (written number)
  "one hundred"     -> 100.0     (compound written number)
  "  $42  "         -> 42.0      (whitespace — strip it)
  "\t$100\t"        -> 100.0     (tab whitespace)
  "-$50"            -> -50.0     (negative amount)
  "-50"             -> -50.0     (negative plain)
  "0"               -> 0.0       (zero)
  "$0"              -> 0.0       (zero with dollar sign)
  "3977"            -> 3977.0    (plain number)
  ".50"             -> 0.5       (leading decimal)
  "$1,234.56"       -> 1234.56   (comma + decimal)

Rules:
- Always strip whitespace first
- Remove $ signs and commas before parsing
- Handle "dollars" and "dollar" suffix
- Support written numbers (zero through ninety, hundred, thousand, million)
- Negative amounts with - prefix must work
- If input truly cannot be parsed, raise ValueError
- Return a float

Output ONLY the raw Python code. No markdown, no backticks, no explanation.
"""


def get_solution() -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": TEACHER_MODEL,
            "prompt": PROMPT_HINT,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "stop": ["```"],
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def to_jsonl_format(solution: str) -> dict:
    """Format for HuggingFace / Colab fine-tuning."""
    return {
        "instruction": "You are an expert Python developer. Return ONLY raw Python code with no markdown, no backticks, no explanation.",
        "input": PROMPT_HINT.strip(),
        "output": solution,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grader_file = TASK_DIR / "grader.py"

    if not grader_file.exists():
        print(f"Grader not found: {grader_file}")
        sys.exit(1)

    print(f"Teacher model: {TEACHER_MODEL}")
    print(f"Target task:   ambigous_spec")
    print(f"Attempts:      {ATTEMPTS}")
    print(f"Min score:     0.8")
    print()

    all_examples = []
    high_quality = []

    for i in range(ATTEMPTS):
        try:
            solution = get_solution()
            scores = run_solution_against_grader(solution, grader_file)
            score = scores.get("score", 0)

            example = {
                "task": "ambigous_spec",
                "prompt": PROMPT_HINT.strip(),
                "solution": solution,
                "scores": scores,
                "score": score,
            }
            all_examples.append(example)

            if score >= 0.8:
                high_quality.append(example)

            status = "✓" if score >= 0.8 else "✗"
            print(f"  [{i+1:02d}/{ATTEMPTS}] {status} score={score:.2f}  "
                  f"({len(high_quality)} high-quality so far)")

        except Exception as e:
            print(f"  [{i+1:02d}/{ATTEMPTS}] ERROR: {e}")

        time.sleep(2.5)

    # Save raw data
    raw_path = OUTPUT_DIR / "training_data.json"
    raw_path.write_text(json.dumps(all_examples, indent=2))

    # Save high quality only
    filtered_path = OUTPUT_DIR / "filtered_data.json"
    filtered_path.write_text(json.dumps(high_quality, indent=2))

    # Save in JSONL format for Colab / HuggingFace
    jsonl_path = OUTPUT_DIR / "train.jsonl"
    with jsonl_path.open("w") as f:
        for ex in high_quality:
            f.write(json.dumps(to_jsonl_format(ex["solution"])) + "\n")

    print(f"\n{'='*50}")
    print(f"Total attempts:        {len(all_examples)}")
    print(f"High quality (>=0.8):  {len(high_quality)}")
    print(f"{'='*50}")
    print(f"Saved:")
    print(f"  Raw data    -> {raw_path}")
    print(f"  Filtered    -> {filtered_path}")
    print(f"  Colab JSONL -> {jsonl_path}")


if __name__ == "__main__":
    main()