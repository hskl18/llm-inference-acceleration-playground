from __future__ import annotations

from importlib import metadata

from llm_accel import __version__
from llm_accel.serving.openai_client import DEFAULT_API_KEY_ENV
from llm_accel.serving.vllm_server import fetch_server_version


def detect_backend_version(backend: str) -> str | None:
    """Backend version from the client host only; prefer resolve_backend_version for real runs."""
    if backend == "mock":
        return f"llm-accel-mock/{__version__}"
    if backend == "vllm":
        try:
            return metadata.version("vllm")
        except metadata.PackageNotFoundError:
            return None
    return None


def resolve_backend_version(
    backend: str,
    base_url: str | None = None,
    *,
    timeout_seconds: float = 5.0,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> tuple[str | None, str]:
    """Return (version, source).

    The serving process, not the client host, decides what served a benchmark, so a remote vLLM
    endpoint is asked through `GET /version` first. The locally installed package is only a
    fallback, and it is wrong whenever the client and the server are different machines.
    """
    if backend == "mock" or (base_url or "").startswith("mock://"):
        return f"llm-accel-mock/{__version__}", "mock"
    if backend == "vllm" and base_url:
        version = fetch_server_version(
            base_url,
            timeout_seconds=timeout_seconds,
            api_key_env=api_key_env,
        )
        if version is not None:
            return version, "server_version_endpoint"
    local = detect_backend_version(backend)
    return (local, "client_package") if local is not None else (None, "unavailable")
