from __future__ import annotations

import platform
import re
import subprocess
import sys
from importlib import import_module
from importlib import metadata
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
    accelerator = collect_accelerator_metadata()
    return {
        "python_version": sys.version.split()[0],
        "operating_system": platform.platform(),
        "git_commit": resolve_git_commit(cwd),
        "hardware_label": hardware_label,
        "gpu_name": gpu_memory.gpu_name if gpu_memory and gpu_memory.available else None,
        **accelerator,
    }


def collect_accelerator_metadata() -> dict[str, str | None]:
    return {
        "gpu_driver_version": _nvidia_smi_field("driver_version"),
        "cuda_version": _torch_cuda_version(),
        "cuda_driver_api_version": _nvidia_smi_cuda_version(),
        "torch_version": _package_version("torch"),
    }


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi_field(field: str) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    return first_line.strip() or None


def _nvidia_smi_cuda_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", completed.stdout)
    return match.group(1) if match else None


def _torch_cuda_version() -> str | None:
    try:
        torch = import_module("torch")
    except (ImportError, OSError):
        return None
    version = getattr(getattr(torch, "version", None), "cuda", None)
    return str(version) if version else None
