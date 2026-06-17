from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    model: str
    backend: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    completed: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunMetadata:
    model: str
    backend: str
    backend_version: str | None
    base_url: str
    api_kind: str
    dtype: str
    quantization: str
    concurrency: int
    input_tokens: int
    output_tokens: int
    request_count: int
    warmup_count: int
    timeout_seconds: float
    timestamp: str
    project_version: str
    python_version: str
    operating_system: str
    git_commit: str | None
    hardware_label: str
    gpu_name: str | None
    workload_mode: str = "synthetic"
    prompt_count: int | None = None
    workload_fingerprint: str | None = None
    shared_prefix_tokens_estimate: int | None = None
    shared_prefix_fingerprint: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
