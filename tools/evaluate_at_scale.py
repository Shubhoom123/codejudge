"""
Evaluates a model against the bulk test suite in data/test_cases.json.
Samples N cases, runs them through the model, and reports accuracy by category.

Usage:
    python tools/evaluate_at_scale.py --model llama3.1 --samples 500
    python tools/evaluate_at_scale.py --model gemma3:12b --samples 1000
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a Python code generator. "
    "Output ONLY the raw Python code. "
    "Do NOT use markdown. Do NOT use backticks. "
    "Do NOT explain anything. Just the code, nothing else."
)

TASK_PROMPT = """Write a function `parse_amount(text: str) -> float` that extracts a monetary amount from a user-typed string.

Examples:
  "$12.50"      -> 12.5
  "100 dollars" -> 100.0
  "fifty"       -> 50.0
  "$1,000"      -> 1000.0

Notes:
- Handle common formats users actually type.
- Return a float.
- If the input cannot be parsed, raise a ValueError."""


def get_solution(model: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": SYSTEM_PROMPT + "\n\n" + TASK_PROMPT,
            "stream": False,
            "options": {"temperature": 0.0},  # greedy — deterministic
        },
        timeout=300,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # Aggressively strip markdown and explanation text
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if code.startswith("python"):
                code = code[len("python"):].lstrip("\n")
            raw = code.strip()
        else:
            raw = parts[1].strip()
    else:
        # No fences — find where the actual code starts
        lines = raw.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("class "):
                in_code = True
            if in_code:
                code_lines.append(line)
        if code_lines:
            raw = "\n".join(code_lines)

    # Truncate at any test code / example code after the main function
    lines = raw.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Stop if we hit test functions or example usage
        if stripped.startswith("def test_") or stripped.startswith("# Test") or stripped.startswith("# Example") or stripped.startswith("if __name__"):
            break
        clean_lines.append(line)
    raw = "\n".join(clean_lines).strip()

    return raw


def run_test_case(solution_code: str, input_text: str, expected) -> tuple[bool, str]:
    """Run a single test case against the solution."""
    code = solution_code + f"""
import json
try:
    result = parse_amount({input_text!r})
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.strip()
        if not output:
            return False, f"No output: {result.stderr[:80]}"
        data = json.loads(output)
        return data["passed"], data["detail"]
    except Exception as e:
        return False, str(e)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of test cases to sample (default: 500)")
    parser.add_argument("--test-cases", default="data/test_cases.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Save detailed results to JSON file")
    args = parser.parse_args()

    # Load test cases
    test_cases_path = Path(args.test_cases)
    if not test_cases_path.exists():
        print(f"Test cases not found at {args.test_cases}")
        print("Run: python tools/generate_test_cases_bulk.py first")
        sys.exit(1)

    all_cases = json.loads(test_cases_path.read_text())
    print(f"Loaded {len(all_cases)} test cases")

    # Sample
    random.seed(args.seed)
    cases = random.sample(all_cases, min(args.samples, len(all_cases)))
    print(f"Sampled {len(cases)} cases for evaluation")

    # Get model solution (once — deterministic, temperature=0)
    print(f"\nQuerying {args.model} for solution...")
    start = time.time()
    try:
        solution = get_solution(args.model)
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
        passed, detail = run_test_case(solution, case["input"], case["expected"])
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
    print(f"\n  By category:")
    print(f"  {'Category':<25} {'Passed':>8} {'Total':>8} {'%':>8}")
    print(f"  {'-'*51}")
    for cat, stats in sorted(by_category.items(), key=lambda x: -x[1]["total"]):
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
        print(f"  {cat:<25} {stats['passed']:>8} {stats['total']:>8} {pct:>7.1f}%")

    # Show failures sample
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n  Sample failures (first 10):")
        for r in failures[:10]:
            print(f"    input={r['input']!r} expected={r['expected']} → {r['detail']}")

    # Save detailed results
    if args.output:
        output_data = {
            "model": args.model,
            "samples": len(cases),
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