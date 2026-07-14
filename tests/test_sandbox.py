"""Tests for sandbox.py — the isolation layer around untrusted model output.

These exist because the harness's main security property (model-generated
code can't read secrets, can't run forever, can't exhaust memory) is easy
to silently break during a refactor. If one of these fails, the sandbox
guarantee is broken.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sandbox import run_isolated


def test_runs_simple_script():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "s.py").write_text("print('hello')")
        result = run_isolated(["python3", "s.py"], cwd=d, timeout=5)
        assert result.stdout.strip() == "hello"


def test_stdout_and_exit_code_are_surfaced():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "s.py").write_text("import sys; sys.exit(3)")
        result = run_isolated(["python3", "s.py"], cwd=d, timeout=5)
        assert result.returncode == 3


def test_wall_clock_timeout_is_enforced():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "s.py").write_text("while True: pass")
        with pytest.raises(subprocess.TimeoutExpired):
            run_isolated(["python3", "s.py"], cwd=d, timeout=1)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "RLIMIT_AS is not reliably enforced by the macOS kernel for large "
        "anonymous allocations the way it is on Linux — this is a real "
        "platform gap in the sandbox, not a test bug. CPU-time and timeout "
        "limits (tested above) still hold on macOS. This test runs for "
        "real in CI, which is Ubuntu — see .github/workflows/ci.yml."
    ),
)
def test_memory_limit_kills_runaway_allocation():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Try to allocate ~2GB; sandbox caps RLIMIT_AS at 256MB, so this
        # should fail (non-zero exit / MemoryError) rather than succeed.
        (d / "s.py").write_text("x = bytearray(2 * 1024 * 1024 * 1024)")
        result = run_isolated(["python3", "s.py"], cwd=d, timeout=5)
        assert result.returncode != 0


def test_env_is_stripped_of_host_secrets():
    os.environ["CODEJUDGE_TEST_SECRET"] = "shhh"
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "s.py").write_text(
                "import os; print(os.environ.get('CODEJUDGE_TEST_SECRET', 'NOT_LEAKED'))"
            )
            result = run_isolated(["python3", "s.py"], cwd=d, timeout=5)
            assert result.stdout.strip() == "NOT_LEAKED"
    finally:
        del os.environ["CODEJUDGE_TEST_SECRET"]
