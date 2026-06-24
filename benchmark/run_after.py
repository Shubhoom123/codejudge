import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from evaluator import run_solution_against_grader, TASKS_DIR

CHECKPOINT_DIR = Path("data/checkpoints/gemma3-codejudge")
BASE_MODEL = "google/gemma-3-4b-it"
OUTPUT_FILE = Path("data/after_results.json")


def get_solution(prompt: str) -> str:
    instruction = (
        "You are an expert Python developer. "
        "Return ONLY raw Python code with no markdown, no backticks, no explanation."
    )
    full_prompt = (
        f"<instruction>{instruction}</instruction>\n"
        f"<task>{prompt.strip()}</task>\n"
        f"<solution>"
    )

    result = subprocess.run(
        [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", BASE_MODEL,
            "--adapter-path", str(CHECKPOINT_DIR),
            "--prompt", full_prompt,
            "--max-tokens", "400",
            "--temp", "0",
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout.strip()

    # Extract content between ========== markers
    if "==========" in output:
        parts = output.split("==========")
        if len(parts) >= 3:
            output = parts[1].strip()
        elif len(parts) == 2:
            output = parts[1].strip()

    # Strip markdown code fences
    lines = output.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    output = "\n".join(lines).strip()

    # Strip solution tags if present
    if "<solution>" in output:
        output = output.split("<solution>")[-1]
    if "</solution>" in output:
        output = output.split("</solution>")[0]

    return output.strip()


def main():
    tasks = [t.name for t in TASKS_DIR.iterdir() if t.is_dir()]
    results = []

    for task_name in tasks:
        task_dir = TASKS_DIR / task_name
        problem_file = task_dir / "problem.md"
        grader_file = task_dir / "grader.py"

        if not problem_file.exists() or not grader_file.exists():
            continue

        print(f"\n  Task: {task_name}")
        prompt = problem_file.read_text()
        solution = get_solution(prompt)
        print(f"  Solution preview: {repr(solution[:120])}")
        scores = run_solution_against_grader(solution, grader_file)
        print(f"  Scores: {scores}")

        results.append({
            "task": task_name,
            "model": "gemma3-finetuned",
            "solution": solution,
            "scores": scores,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nPost-training results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()