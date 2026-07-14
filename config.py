"""Single source of truth for paths, model registry, and prompts.

Previously the Ollama URL, system prompt, and even task-specific example
strings were copy-pasted across evaluator.py, tools/evaluate_at_scale.py,
tools/debug_model.py, and trainer/generate_training_data.py. That let them
drift out of sync. Everything here should be imported, not re-typed.
"""
from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).parent
TASKS_DIR = ROOT_DIR / "tasks"
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

TIMEOUT_SECONDS = 10

# Sent to models before every task prompt. Keeping this in one place means
# changing the output contract (e.g. "no markdown") only requires one edit.
CODE_SYSTEM_PROMPT = (
    "You are a Python code generator. "
    "Output ONLY the raw Python code. "
    "Do NOT use markdown. Do NOT use ``` or ```python. "
    "Do NOT explain anything. Just the code, nothing else."
)

MODELS = {
    "gpt-4o": "openai",
    "llama3.1": "ollama",
    "llama3.2": "ollama",
    "gemma3:12b": "ollama",
    "gemma3:4b": "ollama",
    "mistral": "ollama",
    "codellama": "ollama",
}

DEFAULT_TASK = "ambigous_spec"

# Bulk/scale testing (tools/evaluate_at_scale.py) needs to know the entry
# point function name for each task so it can inject a call into the
# model's solution. Only tasks with a single-argument function are
# supported this way today (see that file's docstring for why).
SCALE_TEST_ENTRY_POINT = {
    "ambigous_spec": "parse_amount",
    "edge_case_trap": "most_frequent",
}


def task_dir(task_name: str) -> Path:
    return TASKS_DIR / task_name


def load_task_prompt(task_name: str) -> str:
    """Read a task's spec straight from its problem.md.

    This is the *only* place a task prompt should be read from. Anything
    that needs the prompt text should call this instead of hardcoding it.
    """
    problem_file = task_dir(task_name) / "problem.md"
    if not problem_file.exists():
        raise FileNotFoundError(f"No problem.md for task '{task_name}'")
    return problem_file.read_text()


def load_task_hint(task_name: str) -> str:
    """Read a task's optional teacher hint (tasks/<task>/hint.md).

    Training-data generation sometimes wants to give the teacher model an
    exhaustive list of formats/edge cases beyond what the (deliberately
    ambiguous) spec says, to produce more consistent training examples.
    Falls back to the plain spec if no hint file exists.
    """
    hint_file = task_dir(task_name) / "hint.md"
    if hint_file.exists():
        return hint_file.read_text()
    return load_task_prompt(task_name)


def load_task_grader(task_name: str) -> Path:
    grader_file = task_dir(task_name) / "grader.py"
    if not grader_file.exists():
        raise FileNotFoundError(f"No grader.py for task '{task_name}'")
    return grader_file


def list_tasks() -> list[str]:
    if not TASKS_DIR.exists():
        return []
    return sorted(t.name for t in TASKS_DIR.iterdir() if t.is_dir())
