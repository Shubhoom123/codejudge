"""Debug script to see exactly what the model outputs and what error occurs."""
import subprocess
import sys
import tempfile
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

response = requests.post(
    OLLAMA_URL,
    json={
        "model": "qwen2-codejudge",
        "prompt": SYSTEM_PROMPT + "\n\n" + TASK_PROMPT,
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

# Apply stripping
if "```" in raw:
    parts = raw.split("```")
    if len(parts) >= 3:
        code = parts[1]
        if code.startswith("python"):
            code = code[len("python"):].lstrip("\n")
        raw = code.strip()
else:
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

print("=== AFTER STRIPPING ===")
print(repr(raw[:500]))
print()
print("=== CODE TO RUN ===")
print(raw[:500])
print()

# Try to run it
test_code = raw + """
import json
try:
    result = parse_amount('$44.28')
    print(json.dumps({"passed": True, "detail": f"Got {result}"}))
except Exception as e:
    print(json.dumps({"passed": False, "detail": str(e)}))
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(test_code)
    tmp = f.name

print("=== RUNNING TEST ===")
result = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=5)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr[:300])
Path(tmp).unlink(missing_ok=True)