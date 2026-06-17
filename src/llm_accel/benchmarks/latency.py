from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from llm_accel import __version__
from llm_accel.metrics.aggregation import summarize_requests
from llm_accel.metrics.environment import collect_environment_metadata
from llm_accel.metrics.io import write_json, write_jsonl, write_request_csv
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.memory import sample_gpu_memory, summarize_memory
from llm_accel.metrics.schemas import RequestMetrics, RunMetadata
from llm_accel.reports.markdown import write_summary_markdown
from llm_accel.reports.plots import write_latency_svg
from llm_accel.serving.openai_client import OpenAICompatibleClient
from llm_accel.serving.versions import detect_backend_version
from llm_accel.workloads.prompts import (
    estimate_prompt_tokens,
    fixed_prompt_batch,
    prompt_fingerprint,
    shared_prefix_fingerprint,
    shared_prefix_tokens,
)
from llm_accel.workloads.synthetic import prompt_batch


def run_latency_benchmark(
    *,
    base_url: str,
    model: str,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    output_dir: str | Path,
    request_count: int = 8,
    warmup_count: int = 0,
    timeout_seconds: float = 120.0,
    dtype: str = "unknown",
    quantization: str = "none",
    backend: str = "openai-compatible",
    seed: int = 42,
    stream: bool = True,
    hardware_label: str = "local",
    api_kind: str = "chat",
    prompt_texts: list[str] | None = None,
) -> dict[str, object]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if prompt_texts is not None and not prompt_texts:
        raise ValueError("prompt_texts must not be empty")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        backend=backend,
        request_timeout_seconds=timeout_seconds,
        api_kind=api_kind,
    )
    memory_before = sample_gpu_memory()
    workload_mode = "fixed_prompts" if prompt_texts is not None else "synthetic"
    workload_fingerprint = prompt_fingerprint(prompt_texts) if prompt_texts is not None else None
    prompt_count = len(prompt_texts) if prompt_texts is not None else None
    shared_tokens = shared_prefix_tokens(prompt_texts) if prompt_texts is not None else None
    shared_fingerprint = shared_prefix_fingerprint(prompt_texts) if prompt_texts is not None else None

    if warmup_count:
        warmup_prompts = fixed_prompt_batch(prompt_texts, warmup_count) if prompt_texts is not None else prompt_batch(warmup_count, input_tokens, seed)
        for index, prompt in enumerate(warmup_prompts):
            client.complete(prompt, output_tokens, index, stream=stream)

    records: list[RequestMetrics] = []
    prompts = fixed_prompt_batch(prompt_texts, request_count) if prompt_texts is not None else prompt_batch(request_count, input_tokens, seed + warmup_count)
    prompt_token_counts = [estimate_prompt_tokens(prompt) for prompt in prompts]
    metadata_input_tokens = round(sum(prompt_token_counts) / len(prompt_token_counts)) if prompt_texts is not None else input_tokens
    effective_backend = "mock" if base_url.startswith("mock://") else backend
    backend_version = detect_backend_version(effective_backend)
    measured_started = time.perf_counter()

    def run_one(index: int, prompt: str) -> RequestMetrics:
        try:
            result = client.complete(prompt, output_tokens, index, stream=stream)
            return RequestMetrics(
                request_id=f"req-{index + 1:06d}",
                model=model,
                backend=effective_backend,
                input_tokens=estimate_prompt_tokens(prompt),
                output_tokens=result.output_tokens,
                concurrency=concurrency,
                ttft_ms=result.ttft_ms,
                tpot_ms=result.tpot_ms,
                total_latency_ms=result.total_latency_ms,
                completed=True,
                error=None,
            )
        except Exception as exc:
            return RequestMetrics(
                request_id=f"req-{index + 1:06d}",
                model=model,
                backend=effective_backend,
                input_tokens=estimate_prompt_tokens(prompt),
                output_tokens=0,
                concurrency=concurrency,
                ttft_ms=0.0,
                tpot_ms=0.0,
                total_latency_ms=0.0,
                completed=False,
                error=str(exc),
            )

    executor = ThreadPoolExecutor(max_workers=concurrency)
    futures = {executor.submit(run_one, index, prompt): index for index, prompt in enumerate(prompts)}
    done, pending = wait(futures, timeout=timeout_seconds)
    for future in done:
        records.append(future.result())
    for future in pending:
        index = futures[future]
        future.cancel()
        records.append(
            RequestMetrics(
                request_id=f"req-{index + 1:06d}",
                model=model,
                backend=effective_backend,
                input_tokens=estimate_prompt_tokens(prompts[index]),
                output_tokens=0,
                concurrency=concurrency,
                ttft_ms=0.0,
                tpot_ms=0.0,
                total_latency_ms=timeout_seconds * 1000,
                completed=False,
                error=f"request timed out after {timeout_seconds} seconds",
            )
        )
    executor.shutdown(wait=False, cancel_futures=True)

    measured_elapsed_seconds = time.perf_counter() - measured_started
    records.sort(key=lambda record: record.request_id)
    memory_after = sample_gpu_memory()
    resolved_config = {
        "base_url": base_url if base_url.startswith(("mock://", "http://localhost", "http://127.0.0.1")) else "redacted",
        "model": model,
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "request_count": request_count,
        "workload_mode": workload_mode,
        "prompt_count": prompt_count,
        "workload_fingerprint": workload_fingerprint,
        "shared_prefix_tokens_estimate": shared_tokens,
        "shared_prefix_fingerprint": shared_fingerprint,
        "warmup_count": warmup_count,
        "timeout_seconds": timeout_seconds,
        "dtype": dtype,
        "quantization": quantization,
        "backend": effective_backend,
        "backend_version": backend_version,
        "api_kind": api_kind,
        "seed": seed,
        "stream": stream,
        "hardware_label": hardware_label,
    }
    environment = collect_environment_metadata(
        cwd=Path.cwd(),
        hardware_label=hardware_label,
        gpu_memory=memory_after if memory_after.available else memory_before,
    )
    metadata = RunMetadata(
        model=model,
        backend=effective_backend,
        backend_version=backend_version,
        base_url=base_url if base_url.startswith(("mock://", "http://localhost", "http://127.0.0.1")) else "redacted",
        api_kind=api_kind,
        dtype=dtype,
        quantization=quantization,
        concurrency=concurrency,
        input_tokens=metadata_input_tokens,
        output_tokens=output_tokens,
        request_count=request_count,
        warmup_count=warmup_count,
        timeout_seconds=timeout_seconds,
        timestamp=datetime.now(timezone.utc).isoformat(),
        project_version=__version__,
        python_version=str(environment["python_version"]),
        operating_system=str(environment["operating_system"]),
        git_commit=environment["git_commit"] if isinstance(environment["git_commit"], str) else None,
        hardware_label=str(environment["hardware_label"]),
        gpu_name=environment["gpu_name"] if isinstance(environment["gpu_name"], str) else None,
        workload_mode=workload_mode,
        prompt_count=prompt_count,
        workload_fingerprint=workload_fingerprint,
        shared_prefix_tokens_estimate=shared_tokens,
        shared_prefix_fingerprint=shared_fingerprint,
    )
    metrics = summarize_requests(records, elapsed_seconds=measured_elapsed_seconds)
    memory = summarize_memory(memory_before, memory_after)
    warnings = _build_run_warnings(
        backend=effective_backend,
        backend_version=backend_version,
        stream=stream,
        memory=memory,
        metrics=metrics,
    )
    summary = {
        "schema_version": metadata.schema_version,
        "metadata": metadata.to_dict(),
        "metrics": metrics,
        "memory": memory,
        "warnings": warnings,
    }
    write_jsonl(out_dir / "raw_requests.jsonl", records)
    write_request_csv(out_dir / "raw_requests.csv", records)
    write_json(out_dir / "resolved_config.json", resolved_config)
    write_json(out_dir / "run_metadata.json", metadata.to_dict())
    write_json(out_dir / "summary.json", summary)
    write_summary_markdown(out_dir / "summary.md", summary)
    write_latency_svg(out_dir / "plots" / "latency.svg", records)
    write_run_manifest(
        out_dir,
        run_type="latency_benchmark",
        artifacts=[
            "manifest.json",
            "resolved_config.json",
            "raw_requests.jsonl",
            "raw_requests.csv",
            "run_metadata.json",
            "summary.json",
            "summary.md",
            "plots/latency.svg",
        ],
    )
    return summary


def _build_run_warnings(
    *,
    backend: str,
    backend_version: str | None,
    stream: bool,
    memory: dict[str, object],
    metrics: dict[str, object],
) -> list[str]:
    warnings: list[str] = []
    if backend == "mock":
        warnings.append("Mock backend results validate workflow only; they are not hardware performance claims.")
    if backend_version is None:
        warnings.append(f"Backend version is unavailable for backend {backend!r}.")
    if not stream:
        warnings.append("Non-streaming mode cannot observe TTFT; TTFT is recorded as total request latency.")
    if not memory.get("available", False):
        after = memory.get("after", {})
        error = after.get("error") if isinstance(after, dict) else None
        suffix = f": {error}" if error else ""
        warnings.append(f"GPU memory telemetry unavailable{suffix}.")
    if int(metrics.get("failed_count", 0)) > 0:
        warnings.append("One or more requests failed; inspect raw_requests.jsonl before interpreting latency or throughput.")
    return warnings
