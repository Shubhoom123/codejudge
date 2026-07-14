import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config
from evaluator import evaluate_task

OUTPUT_FILE = Path("data/before_results.json")


def main():
    tasks = config.list_tasks()
    models = {"gemma3:12b": "ollama"}

    results = []
    for task_name in tasks:
        result = evaluate_task(task_name, models)
        results.append(result)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nBaseline results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()