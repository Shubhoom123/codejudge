import json
from pathlib import Path


INPUT_FILE = Path("data/training_runs/training_data.json")
OUTPUT_FILE = Path("data/training_runs/filtered_data.json")
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
        "instruction": (
            "You are an expert Python developer. "
            "Return ONLY raw Python code with no markdown, no backticks, no explanation."
        ),
        "input": prompt.strip(),
        "output": solution.strip(),
    }


def main():
    raw = json.loads(INPUT_FILE.read_text())
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

    OUTPUT_FILE.write_text(json.dumps(filtered, indent=2))
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()