"""Isolated execution of untrusted, LLM-generated code.

evaluator.py and the scale-testing tools run arbitrary Python that a model
just wrote. It is never imported or exec()'d in-process. It always runs as a
separate subprocess with:

  - a hard wall-clock timeout
  - CPU time and memory limits (POSIX, via resource.setrlimit)
  - a process-count limit, so it can't fork-bomb the host
  - a stripped-down environment (no inherited API keys / secrets)
  - its own temp directory as cwd

This is defense-in-depth, not a full container sandbox: it does not
restrict filesystem access or the network. A model that writes
`requests.get(...)` or `open("/etc/passwd")` can still do so. For a
stronger boundary, run the harness itself inside a container/VM with a
read-only or scratch-only filesystem and no network egress.

Platform note: the memory limit (RLIMIT_AS) is reliably enforced by the
Linux kernel but NOT reliably enforced by macOS for large anonymous
allocations — a runaway `bytearray(...)` that gets killed on Linux may
succeed uninterrupted on macOS. The CPU-time limit, process-count limit,
and wall-clock timeout all still hold on macOS. This is a real gap, not
a documentation nicety — CI runs on Ubuntu specifically so the memory
limit is verified somewhere for real (see tests/test_sandbox.py and
.github/workflows/ci.yml). If you need the memory guarantee locally on
macOS too, run the harness inside Docker/Linux.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CPU_TIME_LIMIT_SECONDS = 5
MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB
PROCESS_LIMIT = 16
WALL_CLOCK_TIMEOUT_SECONDS = 10


def _apply_resource_limits() -> None:
    """Runs inside the child process (POSIX only), right before exec."""
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_CPU, (CPU_TIME_LIMIT_SECONDS, CPU_TIME_LIMIT_SECONDS)
        )
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (PROCESS_LIMIT, PROCESS_LIMIT))
    except (ImportError, ValueError, OSError):
        # `resource` isn't available (e.g. Windows), or the host doesn't
        # permit tightening this limit. Fall back to timeout-only isolation
        # rather than crashing the whole harness.
        pass


def _minimal_env() -> dict[str, str]:
    """A stripped environment so subprocess code can't read API keys, etc."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def run_isolated(
    argv: list[str],
    cwd: Path,
    timeout: int = WALL_CLOCK_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    """Run `argv` as a subprocess with resource limits and a minimal env.

    Raises subprocess.TimeoutExpired if it exceeds `timeout` wall-clock
    seconds. Callers should still treat all output as untrusted.
    """
    preexec = _apply_resource_limits if sys.platform != "win32" else None

    return subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_minimal_env(),
        preexec_fn=preexec,
    )
