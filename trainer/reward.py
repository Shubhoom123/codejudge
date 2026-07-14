import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

# Regression note: this used to be a fixed threshold (score >= 0.8 => reward
# 0.5, included in training). That silently let solutions which failed an
# entire test category through as "high quality" -- e.g. a parse_amount
# that handled every format except written numbers scored exactly 20/24 =
# 0.833, cleared the 0.8 bar, and got trained on with a non-zero reward.
# 77 of the 83 examples (93%) that produced the qwen2-codejudge model in
# Ollama scored between 0.83 and 0.96, not 1.0 -- meaning the training set
# was dominated by near-misses, not perfect examples, which is a plausible
# root cause for the model failing 100% of written_number cases at scale
# despite scoring 83% on the hand-written grader. --min-score now actually
# gates inclusion (it didn't before -- MIN_SCORE was defined but unused)
# and defaults to 1.0: only solutions that pass every single grader test
# make it into the training set by default.
DEFAULT_MIN_SCORE = 1.0


def score_to_reward(score: float, min_score: float) -> float:
    """Reward scales with how close to perfect the solution is, but
    anything below min_score is excluded from training entirely (reward
    0.0), not just down-weighted."""
    if score < min_score:
        return 0.0
    return score


def build_training_prompt(prompt: str, solution: str) -> dict:
    return {
        "instruction": config.CODE_SYSTEM_PROMPT,
        "input": prompt.strip(),
        "output": solution.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=config.DEFAULT_TASK)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                         help=f"Minimum grader score required to keep an example "
                              f"(default: {DEFAULT_MIN_SCORE} -- perfect only). "
                              f"Lowering this reintroduces near-miss solutions "
                              f"into the training set; see the module docstring "
                              f"for why that's risky.")
    args = parser.parse_args()

    run_dir = config.DATA_DIR / "training_runs" / args.task
    input_file = run_dir / "training_data.json"
    output_file = run_dir / "filtered_data.json"
    jsonl_file = run_dir / "train.jsonl"

    if not input_file.exists():
        print(f"No training data at {input_file}. Run trainer/generate_training_data.py --task {args.task} first.")
        sys.exit(1)

    raw = json.loads(input_file.read_text())
    print(f"Loaded {len(raw)} total examples")
    print(f"Minimum score to keep: {args.min_score}")

    filtered = []
    for example in raw:
        reward = score_to_reward(example["score"], args.min_score)
        if reward > 0:
            filtered.append({
                **build_training_prompt(example["prompt"], example["solution"]),
                "reward": reward,
                "task": example["task"],
                "score": example["score"],
            })

    by_task = {}
    for ex in filtered:
        by_task.setdefault(ex["task"], []).append(ex)

    print(f"Filtered to {len(filtered)} high-quality examples")
    for task, examples in by_task.items():
        perfect = sum(1 for e in examples if e["score"] == 1.0)
        print(f"  {task}: {len(examples)} examples ({perfect} perfect)")

    output_file.write_text(json.dumps(filtered, indent=2))

    # train.jsonl is the file that actually gets uploaded to Colab -- this
    # is now the only place it gets written, so there's exactly one
    # filtering decision in the whole pipeline, not two disagreeing ones.
    with jsonl_file.open("w") as f:
        for ex in filtered:
            f.write(json.dumps({
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
            }) + "\n")

    print(f"\nSaved:")
    print(f"  Filtered   -> {output_file}")
    print(f"  Colab JSONL -> {jsonl_file}")

    if not filtered:
        print(
            f"\nWarning: 0 examples cleared --min-score {args.min_score}. "
            f"Either run trainer/generate_training_data.py with more --attempts "
            f"to find perfect solutions, or rerun this script with a lower "
            f"--min-score (accepting the near-miss risk described above)."
        )


if __name__ == "__main__":
    main()
