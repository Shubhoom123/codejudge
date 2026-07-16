import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import openai
import requests

import config
from code_cleaning import clean_llm_code
from sandbox import run_isolated

# Re-exported for backwards compatibility — other modules previously
# imported these names directly from evaluator.py.
MODELS = config.MODELS
TASKS_DIR = config.TASKS_DIR
RESULTS_DIR = config.RESULTS_DIR
TIMEOUT_SECONDS = config.TIMEOUT_SECONDS


def get_available_ollama_models() -> list:
    try:
        response = requests.get(config.OLLAMA_TAGS_URL, timeout=5)
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


def call_openai(prompt: str, model: str) -> str:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": config.CODE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def call_ollama(prompt: str, model: str) -> str:
    # Regression note: this used to pass stop=["```"], meant to trim a
    # trailing closing fence. Ollama's stop tokens match anywhere in the
    # stream, not just at the end -- a model that opens its response with
    # ```python (as qwen2-codejudge-v3 does) gets cut off after emitting
    # those first three backticks, before any real code comes out. That
    # looked like intermittent "Could not import parse_amount" failures
    # with suspiciously fast (<1s) response times, which read like model
    # flakiness but was actually this stop token killing the response on
    # every call where the model happened to open with a fence.
    # code_cleaning.py already strips fences correctly after the full
    # response is in hand, via real parsing, so there's no need for a stop
    # token to do it mid-stream. tools/evaluate_at_scale.py and
    # tools/debug_model.py never set a stop token here, which is why they
    # didn't hit this.
    response = requests.post(
        config.OLLAMA_GENERATE_URL,
        json={
            "model": model,
            "prompt": config.CODE_SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
            "options": {
                "temperature": 0,
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

    return clean_llm_code(raw)


def run_solution_against_grader(solution_code: str, grader_path: Path) -> dict:
    """Run model-generated code against a task's grader, fully isolated.

    Both files are written into a fresh temp directory and the grader is
    executed as a sandboxed subprocess (see sandbox.py) — resource-limited,
    minimal environment, hard timeout — since `solution_code` is untrusted
    LLM output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        solution_file = tmpdir / "solution.py"
        solution_file.write_text(solution_code)

        grader_file = tmpdir / "grader.py"
        grader_file.write_text(grader_path.read_text())

        try:
            result = run_isolated(["python3", "grader.py"], cwd=tmpdir, timeout=TIMEOUT_SECONDS)
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
    return {
        "name": task_name,
        "prompt": config.load_task_prompt(task_name),
        "grader_path": config.load_task_grader(task_name),
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

    tasks_to_run = [args.task] if args.task else config.list_tasks()

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
