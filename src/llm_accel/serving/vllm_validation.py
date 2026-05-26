from __future__ import annotations

import importlib.util
from pathlib import Path

from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.memory import sample_gpu_memory
from llm_accel.serving.health import check_endpoint_health
from llm_accel.serving.openai_client import OpenAICompatibleClient
from llm_accel.serving.vllm import build_vllm_command


def validate_vllm_environment(
    *,
    model: str,
    base_url: str,
    output_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    quantization: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    timeout_seconds: float = 5.0,
    smoke: bool = False,
) -> dict[str, object]:
    command = build_vllm_command(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    import_available = importlib.util.find_spec("vllm") is not None
    gpu_memory = sample_gpu_memory().to_dict()
    endpoint_health = check_endpoint_health(base_url, timeout_seconds=timeout_seconds)
    smoke_result = _run_smoke(base_url, model, timeout_seconds) if smoke else {
        "attempted": False,
        "passed": None,
        "error": None,
    }
    blockers = []
    if not import_available:
        blockers.append("vllm Python package is not importable")
    if not gpu_memory["available"]:
        blockers.append(f"GPU telemetry unavailable: {gpu_memory.get('error')}")
    if not endpoint_health["healthy"]:
        blockers.append(f"endpoint health check failed: {endpoint_health.get('error')}")
    if smoke and not smoke_result["passed"]:
        blockers.append(f"smoke completion failed: {smoke_result.get('error')}")

    report = {
        "model": model,
        "base_url": base_url if base_url.startswith(("mock://", "http://localhost", "http://127.0.0.1")) else "redacted",
        "command": command.to_dict(),
        "checks": {
            "vllm_import_available": import_available,
            "gpu_memory": gpu_memory,
            "endpoint_health": endpoint_health,
            "smoke_completion": smoke_result,
        },
        "ready_for_hardware_benchmark": not blockers,
        "blockers": blockers,
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


def _run_smoke(base_url: str, model: str, timeout_seconds: float) -> dict[str, object]:
    try:
        client = OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            backend="vllm",
            request_timeout_seconds=timeout_seconds,
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
    blockers = report["blockers"]  # type: ignore[index]
    blocker_lines = [f"- {blocker}" for blocker in blockers] or ["- None"]
    text = "\n".join(
        [
            "# vLLM Validation",
            "",
            f"- Model: `{report['model']}`",
            f"- Ready for hardware benchmark: `{report['ready_for_hardware_benchmark']}`",
            f"- vLLM import available: `{checks['vllm_import_available']}`",
            f"- GPU telemetry available: `{gpu['available']}`",
            f"- Endpoint healthy: `{endpoint['healthy']}`",
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
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
