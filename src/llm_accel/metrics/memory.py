from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuMemorySnapshot:
    available: bool
    backend: str
    total_mib: int | None = None
    used_mib: int | None = None
    free_mib: int | None = None
    gpu_name: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_gpu_memory() -> GpuMemorySnapshot:
    if shutil.which("nvidia-smi") is None:
        return GpuMemorySnapshot(
            available=False,
            backend="nvidia-smi",
            error="nvidia-smi not found",
        )

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return GpuMemorySnapshot(
            available=False,
            backend="nvidia-smi",
            error=str(exc),
        )

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 4:
        return GpuMemorySnapshot(
            available=False,
            backend="nvidia-smi",
            error="unexpected nvidia-smi output",
        )

    name, total, used, free = parts
    try:
        return GpuMemorySnapshot(
            available=True,
            backend="nvidia-smi",
            gpu_name=name,
            total_mib=int(total),
            used_mib=int(used),
            free_mib=int(free),
        )
    except ValueError as exc:
        return GpuMemorySnapshot(
            available=False,
            backend="nvidia-smi",
            error=f"could not parse nvidia-smi output: {exc}",
        )


def summarize_memory(before: GpuMemorySnapshot, after: GpuMemorySnapshot) -> dict[str, object]:
    delta_used_mib = None
    if before.used_mib is not None and after.used_mib is not None:
        delta_used_mib = after.used_mib - before.used_mib
    return {
        "before": before.to_dict(),
        "after": after.to_dict(),
        "delta_used_mib": delta_used_mib,
        "available": before.available and after.available,
    }
