import os
import json
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

import openai
import requests

MODELS = {
    "gpt-4o": "openai",
    "llama3.1": "ollama",
    "llama3.2": "ollama",
    "gemma3:12b": "ollama",
    "gemma3:4b": "ollama",
    "mistral": "ollama",
    "codellama": "ollama",
}


def get_available_ollama_models() -> list:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return []


def resolve_model(model_name: str) -> tuple[str, str]:
    """Return (model_name, provider). Auto-detects Ollama models."""
    if model_name in MODELS:
        return model_name, MODELS[model_name]
    # Check if it's available in Ollama
    ollama_models = get_available_ollama_models()
    for om in ollama_models:
        if model_name in om or om.startswith(model_name):
            return om, "ollama"
    raise ValueError(
        f"Unknown model '{model_name}'.\n"
        f"  Known: {list(MODELS.keys())}\n"
        f"  Ollama: {ollama_models}"
    )

TASKS_DIR = Path("tasks")
RESULTS_DIR = Path("results")
TIMEOUT_SECONDS = 10


def call_openai(prompt: str, model: str) -> str:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer. "
                    "Return ONLY raw Python code with no markdown, no backticks, no explanation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def call_ollama(prompt: str, model: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": (
                "You are a Python code generator. "
                "Output ONLY the raw Python code. "
                "Do NOT use markdown. Do NOT use ``` or ```python. "
                "Do NOT explain anything. Just the code, nothing else.\n\n"
                + prompt
            ),
            "stream": False,
            "options": {
                "temperature": 0,
                "stop": ["```"]
            },
        },
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def get_llm_solution(prompt: str, model: str, provider: str) -> str:
    if provider == "openai":
        raw = call_openai(prompt, model)
    elif provider == "ollama":
        raw = call_ollama(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def run_solution_against_grader(solution_code: str, grader_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        solution_file = tmpdir / "solution.py"
        solution_file.write_text(solution_code)

        grader_file = tmpdir / "grader.py"
        grader_file.write_text(grader_path.read_text())

        try:
            result = subprocess.run(
                ["python", "grader.py"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            output = result.stdout.strip()
            stderr = result.stderr.strip()

            if output:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    return {"error": f"Grader output not valid JSON: {output}", "stderr": stderr}
            return {"error": "No output from grader", "stderr": stderr}

        except subprocess.TimeoutExpired:
            return {"error": f"Solution timed out after {TIMEOUT_SECONDS}s"}
        except Exception as e:
            return {"error": str(e)}


def load_task(task_name: str) -> dict:
    task_dir = TASKS_DIR / task_name
    problem_file = task_dir / "problem.md"
    grader_file = task_dir / "grader.py"

    if not problem_file.exists() or not grader_file.exists():
        raise FileNotFoundError(f"Task '{task_name}' missing problem.md or grader.py")

    return {
        "name": task_name,
        "prompt": problem_file.read_text(),
        "grader_path": grader_file,
    }


def evaluate_task(task_name: str, models: dict = MODELS) -> dict:
    print(f"\n{'='*60}")
    print(f"  Task: {task_name}")
    print(f"{'='*60}")

    task = load_task(task_name)
    task_results = {"task": task_name, "models": {}}

    for model, provider in models.items():
        print(f"\n  → Querying {model}...")
        start = time.time()

        try:
            solution = get_llm_solution(task["prompt"], model, provider)
            elapsed = round(time.time() - start, 2)
            print(f"    Solution received in {elapsed}s")

            scores = run_solution_against_grader(solution, task["grader_path"])
            print(f"    Scores: {scores}")

            task_results["models"][model] = {
                "solution": solution,
                "scores": scores,
                "latency_s": elapsed,
            }

        except Exception as e:
            print(f"    ERROR: {e}")
            task_results["models"][model] = {"error": str(e)}

    return task_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CodeJudge: Evaluate LLMs on coding tasks")
    parser.add_argument("--task", type=str, help="Run a specific task by name")
    parser.add_argument("--model", type=str, help="Run only this model (key from MODELS dict)")
    args = parser.parse_args()

    models_to_run = MODELS
    if args.model:
        try:
            model_name, provider = resolve_model(args.model)
            models_to_run = {model_name: provider}
        except ValueError as e:
            print(e)
            return

    tasks_to_run = [args.task] if args.task else [t.name for t in TASKS_DIR.iterdir() if t.is_dir()]

    all_results = []
    for task_name in tasks_to_run:
        result = evaluate_task(task_name, models_to_run)
        all_results.append(result)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"run_{timestamp}.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n\nResults saved to {out_file}")


if __name__ == "__main__":
    main()