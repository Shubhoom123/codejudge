import json
from pathlib import Path

BEFORE_FILE = Path("data/before_results.json")
AFTER_FILE = Path("data/after_results.json")


def get_scores(results: list) -> dict:
    scores = {}
    for entry in results:
        task = entry.get("task") or entry.get("models", {})
        if isinstance(task, str):
            model_data = list(entry.get("models", {}).values())
            if model_data:
                s = model_data[0].get("scores", {}).get("score", 0)
                scores[task] = s
        else:
            scores[entry.get("task", "unknown")] = entry.get("scores", {}).get("score", 0)
    return scores


def main():
    if not BEFORE_FILE.exists() or not AFTER_FILE.exists():
        print("Missing results. Run run_before.py and run_after.py first.")
        return

    before = json.loads(BEFORE_FILE.read_text())
    after = json.loads(AFTER_FILE.read_text())

    before_scores = {}
    for entry in before:
        task = entry["task"]
        model_data = list(entry["models"].values())
        if model_data:
            before_scores[task] = model_data[0].get("scores", {}).get("score", 0)

    after_scores = {}
    for entry in after:
        after_scores[entry["task"]] = entry.get("scores", {}).get("score", 0)

    print("=" * 55)
    print("  BEFORE vs AFTER FINE-TUNING")
    print("=" * 55)
    print(f"\n{'Task':<28} {'Before':>8} {'After':>8} {'Delta':>8}")
    print("-" * 55)

    total_before = 0
    total_after = 0
    count = 0

    for task in sorted(set(list(before_scores.keys()) + list(after_scores.keys()))):
        b = before_scores.get(task, 0)
        a = after_scores.get(task, 0)
        delta = a - b
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        print(f"{task:<28} {b:>7.0%} {a:>7.0%} {arrow} {abs(delta):>5.0%}")
        total_before += b
        total_after += a
        count += 1

    if count:
        avg_before = total_before / count
        avg_after = total_after / count
        avg_delta = avg_after - avg_before
        print("-" * 55)
        print(f"{'AVERAGE':<28} {avg_before:>7.0%} {avg_after:>7.0%}   {avg_delta:>+5.0%}")

    print()


if __name__ == "__main__":
    main()