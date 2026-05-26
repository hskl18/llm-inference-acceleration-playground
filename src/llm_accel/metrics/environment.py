from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from llm_accel.metrics.memory import GpuMemorySnapshot


def resolve_git_commit(cwd: str | Path = ".") -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    commit = completed.stdout.strip()
    return commit or None


def collect_environment_metadata(
    *,
    cwd: str | Path = ".",
    hardware_label: str = "local",
    gpu_memory: GpuMemorySnapshot | None = None,
) -> dict[str, object]:
    return {
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "git_commit": resolve_git_commit(cwd),
        "hardware_label": hardware_label,
        "gpu_name": gpu_memory.gpu_name if gpu_memory and gpu_memory.available else None,
    }
