import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

MIN_SCORE = 0.8


def score_to_reward(score: float) -> float:
    if score == 1.0:
        return 1.0
    elif score >= 0.8:
        return 0.5
    else:
        return 0.0


def build_training_prompt(prompt: str, solution: str) -> dict:
    return {
        "instruction": config.CODE_SYSTEM_PROMPT,
        "input": prompt.strip(),
        "output": solution.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=config.DEFAULT_TASK)
    args = parser.parse_args()

    run_dir = config.DATA_DIR / "training_runs" / args.task
    input_file = run_dir / "training_data.json"
    output_file = run_dir / "filtered_data.json"

    if not input_file.exists():
        print(f"No training data at {input_file}. Run trainer/generate_training_data.py --task {args.task} first.")
        sys.exit(1)

    raw = json.loads(input_file.read_text())
    print(f"Loaded {len(raw)} total examples")

    filtered = []
    for example in raw:
        reward = score_to_reward(example["score"])
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
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
