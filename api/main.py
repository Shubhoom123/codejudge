"""FastAPI wrapper around the CodeJudge evaluation harness.

Exposes evaluator.py / analyze.py's logic over HTTP instead of only the
CLI, so it can be driven from a dashboard or another service. /evaluate
still needs a real Ollama instance reachable at config.OLLAMA_GENERATE_URL
— this is a thin layer over the existing harness, not a new inference
backend.

Run:
    pip install -r api/requirements.txt
    uvicorn api.main:app --reload

Docs: http://localhost:8000/docs

Security note: /evaluate triggers model inference *and* sandboxed
execution of whatever code the model returns (see sandbox.py for the
isolation it gets). That's an expensive, code-executing endpoint — if
this is ever exposed beyond localhost, put auth and rate limiting in
front of it. Neither is included here since it's meant to run locally
alongside Ollama.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))
import config
from evaluator import evaluate_task, resolve_model

app = FastAPI(
    title="CodeJudge API",
    description="Evaluate LLM code generation quality against automated graders.",
    version="1.0.0",
)


class EvaluateRequest(BaseModel):
    model: str
    task: str = config.DEFAULT_TASK


class TaskInfo(BaseModel):
    name: str
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks", response_model=list[str])
def list_tasks():
    return config.list_tasks()


@app.get("/tasks/{task_name}", response_model=TaskInfo)
def get_task(task_name: str):
    try:
        prompt = config.load_task_prompt(task_name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    return TaskInfo(name=task_name, prompt=prompt)


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    if req.task not in config.list_tasks():
        raise HTTPException(status_code=404, detail=f"Task '{req.task}' not found")

    try:
        model_name, provider = resolve_model(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return evaluate_task(req.task, {model_name: provider})


@app.get("/results")
def list_results():
    if not config.RESULTS_DIR.exists():
        return []
    return sorted(f.name for f in config.RESULTS_DIR.glob("run_*.json"))


@app.get("/results/latest")
def latest_result():
    files = sorted(config.RESULTS_DIR.glob("run_*.json")) if config.RESULTS_DIR.exists() else []
    if not files:
        raise HTTPException(status_code=404, detail="No results yet")
    return json.loads(files[-1].read_text())


@app.get("/results/{filename}")
def get_result(filename: str):
    # filename comes straight from the URL — validate it against the
    # exact shape we generate (run_<timestamp>.json) before touching the
    # filesystem, so a crafted filename like "../../etc/passwd" can't
    # escape RESULTS_DIR.
    if not filename.startswith("run_") or not filename.endswith(".json") or "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid result filename")

    path = config.RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return json.loads(path.read_text())
