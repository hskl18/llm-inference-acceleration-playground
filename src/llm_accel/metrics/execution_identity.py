from __future__ import annotations

import hashlib


def endpoint_sha256(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def displayed_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.startswith(("mock://", "http://localhost", "http://127.0.0.1")):
        return normalized
    return "redacted"


def execution_identity(
    *,
    profile: str,
    model: str,
    backend: str,
    base_url: str,
) -> dict[str, str]:
    effective_backend = "mock" if base_url.startswith("mock://") else backend
    return {
        "profile": profile,
        "model": model,
        "backend": effective_backend,
        "base_url": displayed_base_url(base_url),
        "endpoint_sha256": endpoint_sha256(base_url),
    }
