from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SCHEMA_VERSION = "0.2"


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
    started_offset_ms: float = 0.0
    completed_offset_ms: float = 0.0
    scheduled_offset_ms: float = 0.0
    dispatch_offset_ms: float = 0.0
    queue_delay_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0

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
    model_revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    optimization_profile: str = "baseline"
    gpu_driver_version: str | None = None
    cuda_version: str | None = None
    cuda_driver_api_version: str | None = None
    torch_version: str | None = None
    server_command_sha256: str | None = None
    stream: bool = True
    request_schedule: str = "closed-loop"
    request_rate_rps: float | None = None
    client_processes: int = 1
    client_workers: int = 1
    queue_delay_warning_ms: float = 10.0
    client_configuration: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
