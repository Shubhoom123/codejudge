"""Debug script to see exactly what the model outputs and what error occurs."""
import argparse
import sys
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
import config
from code_cleaning import clean_llm_code
from sandbox import run_isolated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2-codejudge")
    parser.add_argument("--task", default=config.DEFAULT_TASK)
    parser.add_argument("--input", default="$44.28", help="Sample input to test the entry point with")
    args = parser.parse_args()

    prompt = config.load_task_prompt(args.task)

    response = requests.post(
        config.OLLAMA_GENERATE_URL,
        json={
            "model": args.model,
            "prompt": config.CODE_SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=300,
    )

    raw = response.json()["response"].strip()
    print("=== RAW OUTPUT ===")
    print(repr(raw[:500]))
    print()
    print("=== VISIBLE OUTPUT ===")
    print(raw[:500])
    print()

    # Same cleanup evaluator.py and evaluate_at_scale.py apply: strip
    # markdown fences, then drop trailing self-test/example code (a
    # common failure mode — the model appends a self-test with an
    # undefined name, which crashes on import and masks an otherwise
    # correct solution as a total failure).
    raw = clean_llm_code(raw)

    print("=== AFTER STRIPPING ===")
    print(repr(raw[:500]))
    print()
    print("=== CODE TO RUN ===")
    print(raw[:500])
    print()

    entry_point = config.SCALE_TEST_ENTRY_POINT.get(args.task, "parse_amount")
    test_code = raw + f"""
import json
try:
    result = {entry_point}({args.input!r})
    print(json.dumps({{"passed": True, "detail": f"Got {{result}}"}}))
except Exception as e:
    print(json.dumps({{"passed": False, "detail": str(e)}}))
"""

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script = tmp_path / "check.py"
        script.write_text(test_code)

        print("=== RUNNING TEST ===")
        result = run_isolated(["python3", "check.py"], cwd=tmp_path, timeout=5)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr[:300])


if __name__ == "__main__":
    main()
