from __future__ import annotations

from importlib import metadata

from llm_accel import __version__


def detect_backend_version(backend: str) -> str | None:
    if backend == "mock":
        return f"llm-accel-mock/{__version__}"
    if backend == "vllm":
        try:
            return metadata.version("vllm")
        except metadata.PackageNotFoundError:
            return None
    return None
