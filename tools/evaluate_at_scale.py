"""
Evaluates a model against the bulk test suite in data/test_cases.json.
Samples N cases, runs them through the model, and reports accuracy by category.

Scope: this only supports tasks whose entry point is a single-argument
function returning a plain value — see config.SCALE_TEST_ENTRY_POINT.
`race_condition` (a stateful Counter class) isn't a fit for this bulk,
input->expected style of testing; use evaluator.py for that task instead,
which runs its grader.py directly.

Usage:
    python tools/evaluate_at_scale.py --model llama3.1 --samples 500
    python tools/evaluate_at_scale.py --model gemma3:12b --samples 1000 --task ambigous_spec
"""

import argparse
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).parent.parent))
import config
from code_cleaning import clean_llm_code
from sandbox import run_isolated
from test_case_store import load_manifest, reservoir_sample


def get_solution(model: str, task: str, seed: int | None = None) -> str:
    prompt = config.load_task_prompt(task)
    options = {"temperature": 0.0}  # greedy — deterministic
    if seed is not None:
        options["seed"] = seed

    response = requests.post(
        config.OLLAMA_GENERATE_URL,
        json={
            "model": model,
            "prompt": config.CODE_SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
            "options": options,
        },
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    if "```" not in raw:
        # No fences — the model may have prefixed prose before the code;
        # find where an import/def/class actually starts.
        lines = raw.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "import ", "from ", "class ")):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            raw = "\n".join(code_lines)

    # Shared cleanup: strip fences, then drop trailing self-test/example
    # code via real AST parsing (see code_cleaning.py for why this matters
    # more than it sounds — a trailing self-test with an undefined name
    # can crash the whole grading run).
    return clean_llm_code(raw)


def run_test_case(solution_code: str, entry_point: str, input_text: str, expected) -> tuple[bool, str]:
    """Run a single test case against the solution, sandboxed."""
    code = solution_code + f"""
import json
try:
    result = {entry_point}({input_text!r})
    expected = {expected!r}
    if expected == "raises_ValueError":
        print(json.dumps({{"passed": False, "detail": f"Expected ValueError but got {{result}}"}}))
    else:
        if abs(result - float(expected)) < 0.001:
            print(json.dumps({{"passed": True, "detail": f"Got {{result}}"}}))
        else:
            print(json.dumps({{"passed": False, "detail": f"Expected {{expected}}, got {{result}}"}}))
except ValueError as e:
    expected = {expected!r}
    if expected == "raises_ValueError":
        print(json.dumps({{"passed": True, "detail": "Correctly raised ValueError"}}))
    else:
        print(json.dumps({{"passed": False, "detail": f"Raised ValueError: {{e}}"}}))
except Exception as e:
    print(json.dumps({{"passed": False, "detail": f"Error: {{type(e).__name__}}: {{e}}"}}))
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        script = tmp_path / "check.py"
        script.write_text(code)

        try:
            result = run_isolated(["python3", "check.py"], cwd=tmp_path, timeout=5)
            output = result.stdout.strip()
            if not output:
                return False, f"No output: {result.stderr[:80]}"
            data = json.loads(output)
            return data["passed"], data["detail"]
        except Exception as e:
            return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default=config.DEFAULT_TASK,
                        help=f"Task to evaluate (default: {config.DEFAULT_TASK}). "
                             f"Supported: {list(config.SCALE_TEST_ENTRY_POINT)}")
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of test cases to sample (default: 500)")
    parser.add_argument("--test-cases", default="data/test_cases_bulk",
                        help="Either a sharded corpus directory (from "
                             "generate_test_cases_bulk.py) or a legacy single "
                             "test_cases.json file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for sampling AND, where the backend supports it, "
                             "for the model's generation — for reproducible runs")
    parser.add_argument("--output", default=None,
                        help="Save detailed results to JSON file")
    args = parser.parse_args()

    if args.task not in config.SCALE_TEST_ENTRY_POINT:
        print(f"Task '{args.task}' isn't supported for scale testing.")
        print(f"Supported tasks: {list(config.SCALE_TEST_ENTRY_POINT)}")
        print("For other tasks, use evaluator.py --task <name> instead.")
        sys.exit(1)
    entry_point = config.SCALE_TEST_ENTRY_POINT[args.task]

    # Load test cases. A sharded corpus directory can be arbitrarily large
    # (millions of cases, hundreds of MB) — reservoir_sample streams
    # through it and never holds more than `samples` cases in memory. A
    # legacy single JSON file is loaded whole, same as before.
    test_cases_path = Path(args.test_cases)
    if not test_cases_path.exists():
        print(f"Test cases not found at {args.test_cases}")
        print("Run: python tools/generate_test_cases_bulk.py first")
        sys.exit(1)

    if test_cases_path.is_dir():
        manifest = load_manifest(test_cases_path)
        total = manifest.get("total_cases", "unknown")
        print(f"Corpus at {test_cases_path} has {total:,} cases" if isinstance(total, int)
              else f"Corpus at {test_cases_path} (no manifest found)")
        cases = reservoir_sample(test_cases_path, k=args.samples, seed=args.seed)
    else:
        all_cases = json.loads(test_cases_path.read_text())
        print(f"Loaded {len(all_cases)} test cases")
        random.seed(args.seed)
        cases = random.sample(all_cases, min(args.samples, len(all_cases)))

    print(f"Sampled {len(cases)} cases for evaluation")

    # Get model solution (once — deterministic, temperature=0)
    print(f"\nQuerying {args.model} for a '{args.task}' solution...")
    start = time.time()
    try:
        solution = get_solution(args.model, args.task, seed=args.seed)
    except Exception as e:
        print(f"Failed to get solution: {e}")
        sys.exit(1)
    elapsed = time.time() - start
    print(f"Solution received in {elapsed:.1f}s")
    print(f"Preview: {solution[:120]!r}...")

    # Run all test cases
    print(f"\nRunning {len(cases)} test cases...")
    results = []
    by_category = defaultdict(lambda: {"passed": 0, "total": 0})

    for i, case in enumerate(cases):
        passed, detail = run_test_case(solution, entry_point, case["input"], case["expected"])
        cat = case.get("category", "unknown")
        by_category[cat]["total"] += 1
        if passed:
            by_category[cat]["passed"] += 1

        results.append({
            "input": case["input"],
            "expected": case["expected"],
            "category": cat,
            "passed": passed,
            "detail": detail,
        })

        if (i + 1) % 50 == 0:
            done = sum(1 for r in results if r["passed"])
            pct = done / len(results) * 100
            print(f"  [{i+1}/{len(cases)}] Running accuracy: {pct:.1f}%")

    # Summary
    total_passed = sum(1 for r in results if r["passed"])
    total = len(results)
    overall = total_passed / total * 100

    print(f"\n{'='*55}")
    print(f"  RESULTS: {args.model}")
    print(f"{'='*55}")
    print(f"  Overall: {total_passed}/{total} = {overall:.1f}%")
    print("\n  By category:")
    print(f"  {'Category':<25} {'Passed':>8} {'Total':>8} {'%':>8}")
    print(f"  {'-'*51}")
    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]["total"]):
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {cat:<25} {stats['passed']:>8} {stats['total']:>8} {pct:>7.1f}%")

    # Show failures sample
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n  Sample failures (first 10):")
        for r in failures[:10]:
            print(f"    input={r['input']!r} expected={r['expected']} → {r['detail']}")

    # Save detailed results
    if args.output:
        output_data = {
            "model": args.model,
            "task": args.task,
            "samples": len(cases),
            "seed": args.seed,
            "overall_accuracy": round(overall / 100, 4),
            "by_category": {
                cat: {
                    "passed": s["passed"],
                    "total": s["total"],
                    "accuracy": round(s["passed"] / s["total"], 4) if s["total"] else 0
                }
                for cat, s in by_category.items()
            },
            "results": results,
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\n  Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
