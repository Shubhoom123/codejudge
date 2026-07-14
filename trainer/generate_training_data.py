"""
Uses a strong teacher model to generate high-quality training data for a
task, by repeatedly sampling solutions and keeping the ones that score
well against that task's grader.py.

Originally hardcoded to ambigous_spec/parse_amount; now takes --task so
edge_case_trap and race_condition can go through the same pipeline.

Output: data/training_runs/<task>/training_data.json
        data/training_runs/<task>/train.jsonl  (Colab/HuggingFace format)

Note on reproducibility: teacher sampling uses temperature=0.8 on purpose
(diversity of solutions increases the odds of a variety of high-scoring
ones to train on). That makes individual runs non-deterministic. --seed
is passed to Ollama's `options.seed` where the backend/model supports it,
which makes runs *more* reproducible but is not a hard guarantee across
all Ollama versions/quantizations.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
import config
from evaluator import run_solution_against_grader

DEFAULT_ATTEMPTS = 100
DEFAULT_MIN_SCORE = 0.8


def get_solution(teacher_model: str, prompt: str, seed: int | None) -> str:
    options = {"temperature": 0.8, "stop": ["```"]}
    if seed is not None:
        options["seed"] = seed

    response = requests.post(
        config.OLLAMA_GENERATE_URL,
        json={
            "model": teacher_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
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


def to_jsonl_format(prompt: str, solution: str) -> dict:
    """Format for HuggingFace / Colab fine-tuning."""
    return {
        "instruction": config.CODE_SYSTEM_PROMPT,
        "input": prompt.strip(),
        "output": solution,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=config.DEFAULT_TASK,
                        help=f"Task to generate training data for (default: {config.DEFAULT_TASK})")
    parser.add_argument("--teacher-model", default="gemma3:12b")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--seed", type=int, default=None,
                         help="Passed to the Ollama sampler for more reproducible runs")
    args = parser.parse_args()

    output_dir = config.DATA_DIR / "training_runs" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        grader_file = config.load_task_grader(args.task)
        prompt = config.load_task_hint(args.task)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"Teacher model: {args.teacher_model}")
    print(f"Target task:   {args.task}")
    print(f"Attempts:      {args.attempts}")
    print(f"Min score:     {args.min_score}")
    print(f"Seed:          {args.seed}")
    print()

    all_examples = []
    high_quality = []

    for i in range(args.attempts):
        try:
            solution = get_solution(args.teacher_model, prompt, args.seed)
            scores = run_solution_against_grader(solution, grader_file)
            score = scores.get("score", 0)

            example = {
                "task": args.task,
                "prompt": prompt.strip(),
                "solution": solution,
                "scores": scores,
                "score": score,
            }
            all_examples.append(example)

            if score >= args.min_score:
                high_quality.append(example)

            status = "✓" if score >= args.min_score else "✗"
            print(f"  [{i+1:02d}/{args.attempts}] {status} score={score:.2f}  "
                  f"({len(high_quality)} high-quality so far)")

        except Exception as e:
            print(f"  [{i+1:02d}/{args.attempts}] ERROR: {e}")

        time.sleep(2.5)

    # Save raw data
    raw_path = output_dir / "training_data.json"
    raw_path.write_text(json.dumps(all_examples, indent=2))

    # Save high quality only
    filtered_path = output_dir / "filtered_data.json"
    filtered_path.write_text(json.dumps(high_quality, indent=2))

    # Save in JSONL format for Colab / HuggingFace
    jsonl_path = output_dir / "train.jsonl"
    with jsonl_path.open("w") as f:
        for ex in high_quality:
            f.write(json.dumps(to_jsonl_format(prompt, ex["solution"])) + "\n")

    print(f"\n{'='*50}")
    print(f"Total attempts:        {len(all_examples)}")
    print(f"High quality (>={args.min_score}):  {len(high_quality)}")
    print(f"{'='*50}")
    print("Saved:")
    print(f"  Raw data    -> {raw_path}")
    print(f"  Filtered    -> {filtered_path}")
    print(f"  Colab JSONL -> {jsonl_path}")


if __name__ == "__main__":
    main()
