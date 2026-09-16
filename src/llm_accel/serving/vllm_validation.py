from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_accel.metrics.execution_identity import displayed_base_url
from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.memory import sample_gpu_memory
from llm_accel.serving.health import check_endpoint_health
from llm_accel.serving.openai_client import DEFAULT_API_KEY_ENV, OpenAICompatibleClient
from llm_accel.serving.versions import resolve_backend_version
from llm_accel.serving.vllm import build_vllm_command
from llm_accel.serving.vllm_server import fetch_serving_state


def validate_vllm_environment(
    *,
    model: str,
    base_url: str,
    output_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    revision: str | None = None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    quantization: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    speculative_method: str | None = None,
    speculative_model: str | None = None,
    num_speculative_tokens: int | None = None,
    timeout_seconds: float = 5.0,
    smoke: bool = False,
    same_host: bool = False,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    """Validate the serving endpoint, and the client host only when the user declares a same-host run.

    A benchmark client usually runs on a different machine than the server, so a local `vllm`
    import and a local `nvidia-smi` say nothing about what serves the endpoint. Backend identity
    comes from the server's `/version` endpoint and serving state from its Prometheus `/metrics`.
    """
    command = build_vllm_command(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        revision=revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        speculative_method=speculative_method,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
    )
    import_available = importlib.util.find_spec("vllm") is not None
    gpu_memory = sample_gpu_memory().to_dict()
    endpoint_health = check_endpoint_health(
        base_url,
        timeout_seconds=timeout_seconds,
        api_key_env=api_key_env,
    )
    backend_version, backend_version_source = resolve_backend_version(
        "vllm",
        base_url,
        timeout_seconds=timeout_seconds,
        api_key_env=api_key_env,
    )
    serving_state = (
        {"available": False, "error": "mock endpoint", "counters": {}, "acceptance": None, "prefix_cache_hit_rate": None}
        if base_url.startswith("mock://")
        else fetch_serving_state(base_url, timeout_seconds=timeout_seconds, api_key_env=api_key_env)
    )
    smoke_result = _run_smoke(base_url, model, timeout_seconds, api_key_env) if smoke else {
        "attempted": False,
        "passed": None,
        "error": None,
    }

    blockers: list[str] = []
    warnings: list[str] = []
    if not endpoint_health["healthy"]:
        blockers.append(
            f"endpoint health check {endpoint_health.get('status')}: {endpoint_health.get('error')}"
        )
    if backend_version_source != "server_version_endpoint":
        blockers.append(
            "server /version did not report a backend version; the serving process cannot be identified"
        )
    if not serving_state["available"]:
        warnings.append(
            f"server /metrics is unavailable ({serving_state.get('error')}); "
            "serving state such as speculative acceptance cannot be recorded"
        )
    if smoke and not smoke_result["passed"]:
        blockers.append(f"smoke completion failed: {smoke_result.get('error')}")

    host_checks = [
        (import_available, "vllm Python package is not importable on this client"),
        (bool(gpu_memory["available"]), f"GPU telemetry unavailable on this client: {gpu_memory.get('error')}"),
    ]
    for passed, message in host_checks:
        if passed:
            continue
        if same_host:
            blockers.append(message)
        else:
            warnings.append(f"{message} (informational: this run is not declared same-host)")

    report = {
        "model": model,
        "base_url": displayed_base_url(base_url),
        "same_host": same_host,
        "command": command.to_dict(),
        "checks": {
            "backend_version": backend_version,
            "backend_version_source": backend_version_source,
            "serving_state": serving_state,
            "vllm_import_available": import_available,
            "gpu_memory": gpu_memory,
            "endpoint_health": endpoint_health,
            "smoke_completion": smoke_result,
        },
        "ready_for_hardware_benchmark": not blockers,
        "blockers": blockers,
        "warnings": warnings,
    }
    out_dir = Path(output_dir)
    write_json(out_dir / "vllm_validation.json", report)
    _write_markdown(out_dir / "vllm_validation.md", report)
    write_run_manifest(
        out_dir,
        run_type="vllm_validation",
        artifacts=["manifest.json", "vllm_validation.json", "vllm_validation.md"],
    )
    return report


def _run_smoke(
    base_url: str,
    model: str,
    timeout_seconds: float,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    try:
        client = OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            backend="vllm",
            request_timeout_seconds=timeout_seconds,
            api_key_env=api_key_env,
        )
        result = client.complete("Say ready.", max_tokens=8, stream=True)
        return {
            "attempted": True,
            "passed": bool(result.output_text.strip()),
            "output_tokens": result.output_tokens,
            "ttft_ms": result.ttft_ms,
            "total_latency_ms": result.total_latency_ms,
            "error": None,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "passed": False,
            "output_tokens": 0,
            "ttft_ms": 0.0,
            "total_latency_ms": 0.0,
            "error": str(exc),
        }


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    checks = report["checks"]  # type: ignore[index]
    gpu = checks["gpu_memory"]  # type: ignore[index]
    endpoint = checks["endpoint_health"]  # type: ignore[index]
    smoke = checks["smoke_completion"]  # type: ignore[index]
    serving_state = checks["serving_state"]  # type: ignore[index]
    blocker_lines = [f"- {blocker}" for blocker in report["blockers"]] or ["- None"]  # type: ignore[index]
    warning_lines = [f"- {warning}" for warning in report["warnings"]] or ["- None"]  # type: ignore[index]
    text = "\n".join(
        [
            "# vLLM Validation",
            "",
            f"- Model: `{report['model']}`",
            f"- Ready for hardware benchmark: `{report['ready_for_hardware_benchmark']}`",
            f"- Declared same-host run: `{report['same_host']}`",
            f"- Server backend version: `{checks['backend_version']}` (source `{checks['backend_version_source']}`)",
            f"- Server /metrics available: `{serving_state['available']}`",
            f"- Client vLLM import available: `{checks['vllm_import_available']}`",
            f"- Client GPU telemetry available: `{gpu['available']}`",
            f"- Endpoint status: `{endpoint['status']}`",
            f"- Smoke attempted: `{smoke['attempted']}`",
            "",
            "## Startup Command",
            "",
            "```bash",
            str(report["command"]["shell_command"]),  # type: ignore[index]
            "```",
            "",
            "## Blockers",
            "",
            *blocker_lines,
            "",
            "## Warnings",
            "",
            *warning_lines,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, text)
