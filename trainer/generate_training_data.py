"""
Uses a strong teacher model to generate candidate training data for a
task, by repeatedly sampling solutions and grading every attempt against
that task's grader.py.

Originally hardcoded to ambigous_spec/parse_amount; now takes --task so
edge_case_trap and race_condition can go through the same pipeline.

Output: data/training_runs/<task>/training_data.json  (every attempt, with its score)

This script deliberately does NOT filter or produce train.jsonl itself
anymore. It used to, with its own separate min-score threshold (0.8) --
which silently duplicated and disagreed with trainer/reward.py's
filtering logic. That let solutions which failed an entire grader
category (e.g. every written-number test) through as "high quality"
training data, because 0.8 is easy to clear by acing everything else and
missing one category. Run trainer/reward.py next -- it's the single
place filtering decisions get made now.

Second regression fixed here: the teacher is queried with hint.md (a
fully-enumerated spec) so it reliably produces correct solutions, but
that same hint.md text used to also get stored as the training
example's "prompt" -- meaning the student was fine-tuned to respond to
hint.md's wording specifically. evaluator.py evaluates the student
against problem.md (a deliberately shorter, ambiguous spec), a
different document entirely. A LoRA fine-tune on a few dozen examples
generalizes narrowly; training the student against one prompt and
evaluating it against a differently-worded one is enough to make a real
fine-tune look like it had zero effect, which is exactly what happened
(a v2 model scored an identical 20/24 to v1, failing the identical four
written-number tests). Grading doesn't care which prompt text is
attached to a solution -- run_solution_against_grader() only sees the
solution code -- so it's safe to query the teacher with hint.md while
recording the training example against problem.md instead.

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
# Only used for the live "✓/✗ ... so far" progress display below, not for
# filtering what gets written to disk -- trainer/reward.py owns that now.
PROGRESS_DISPLAY_THRESHOLD = 1.0


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=config.DEFAULT_TASK,
                        help=f"Task to generate training data for (default: {config.DEFAULT_TASK})")
    parser.add_argument("--teacher-model", default="gemma3:12b")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=None,
                         help="Passed to the Ollama sampler for more reproducible runs")
    args = parser.parse_args()

    output_dir = config.DATA_DIR / "training_runs" / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        grader_file = config.load_task_grader(args.task)
        # teacher_prompt: what the teacher is asked to solve (rich, exhaustive --
        #   maximizes the odds of a fully-correct solution).
        # student_prompt: what actually gets recorded as the training example's
        #   input, matching the shorter, ambiguous prompt evaluator.py sends
        #   the student at evaluation time. These are deliberately different
        #   documents -- see the module docstring for why conflating them is
        #   a real bug, not a simplification.
        teacher_prompt = config.load_task_hint(args.task)
        student_prompt = config.load_task_prompt(args.task)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    print(f"Teacher model: {args.teacher_model}")
    print(f"Target task:   {args.task}")
    print(f"Attempts:      {args.attempts}")
    print(f"Seed:          {args.seed}")
    print()

    all_examples = []
    perfect_so_far = 0

    for i in range(args.attempts):
        try:
            solution = get_solution(args.teacher_model, teacher_prompt, args.seed)
            scores = run_solution_against_grader(solution, grader_file)
            score = scores.get("score", 0)

            example = {
                "task": args.task,
                "prompt": student_prompt.strip(),
                "solution": solution,
                "scores": scores,
                "score": score,
            }
            all_examples.append(example)

            if score >= PROGRESS_DISPLAY_THRESHOLD:
                perfect_so_far += 1

            status = "✓" if score >= PROGRESS_DISPLAY_THRESHOLD else "✗"
            print(f"  [{i+1:02d}/{args.attempts}] {status} score={score:.2f}  "
                  f"({perfect_so_far} perfect so far)")

        except Exception as e:
            print(f"  [{i+1:02d}/{args.attempts}] ERROR: {e}")

        time.sleep(2.5)

    raw_path = output_dir / "training_data.json"
    raw_path.write_text(json.dumps(all_examples, indent=2))

    print(f"\n{'='*50}")
    print(f"Total attempts:  {len(all_examples)}")
    print(f"Perfect scores:  {perfect_so_far}")
    print(f"{'='*50}")
    print(f"Saved: {raw_path}")
    print(f"\nNext: python trainer/reward.py --task {args.task}")
    print("(that's the step that filters this and writes train.jsonl)")


if __name__ == "__main__":
    main()
