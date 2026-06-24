import json
import argparse
from pathlib import Path

RESULTS_DIR = Path("results")


def load_latest() -> list:
    files = sorted(RESULTS_DIR.glob("run_*.json"))
    if not files:
        raise FileNotFoundError("No result files found. Run evaluator.py first.")
    latest = files[-1]
    print(f"Loading: {latest}\n")
    return json.loads(latest.read_text())


def print_summary(results: list):
    print("=" * 65)
    print("  CODEJUDGE RESULTS SUMMARY")
    print("=" * 65)

    all_models = set()
    for task_result in results:
        all_models.update(task_result["models"].keys())
    all_models = sorted(all_models)

    header = f"{'Task':<28}" + "".join(f"{m[:16]:<18}" for m in all_models)
    print(f"\n{header}")
    print("-" * len(header))

    for task_result in results:
        task = task_result["task"]
        row = f"{task:<28}"
        for model in all_models:
            model_data = task_result["models"].get(model, {})
            if "error" in model_data and "scores" not in model_data:
                row += f"{'ERROR':<18}"
            else:
                scores = model_data.get("scores", {})
                p = scores.get("passed", "?")
                t = scores.get("total", "?")
                s = scores.get("score", 0)
                row += f"{p}/{t} ({s:.0%})       "[:18]
        print(row)

    print(f"\n\n{'=' * 65}")
    print("  FAILURE PATTERN ANALYSIS")
    print("=" * 65)

    for task_result in results:
        task = task_result["task"]
        print(f"\n  Task: {task}")
        print(f"  {'-' * 40}")

        test_failures: dict[str, list[str]] = {}

        for model, model_data in task_result["models"].items():
            scores = model_data.get("scores", {})
            for test in scores.get("tests", []):
                if not test["passed"]:
                    test_failures.setdefault(test["name"], []).append(
                        f"{model}: {test['detail']}"
                    )

        if not test_failures:
            print("  ✓ All models passed all tests.")
        else:
            for test_name, failures in test_failures.items():
                print(f"\n  ✗ {test_name}")
                for f in failures:
                    print(f"      {f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Specific result file to analyze")
    args = parser.parse_args()

    if args.file:
        results = json.loads(Path(args.file).read_text())
    else:
        results = load_latest()

    print_summary(results)


if __name__ == "__main__":
    main()