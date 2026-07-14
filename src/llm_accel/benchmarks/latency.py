from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import queue
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from llm_accel import __version__
from llm_accel.metrics.aggregation import summarize_requests
from llm_accel.metrics.environment import collect_environment_metadata, environment_fingerprint
from llm_accel.metrics.execution_identity import displayed_base_url, endpoint_sha256
from llm_accel.metrics.io import write_bytes_atomic, write_json, write_jsonl, write_request_csv
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.memory import sample_gpu_memory, summarize_memory
from llm_accel.metrics.schemas import RequestMetrics, RunMetadata
from llm_accel.metrics.token_counting import (
    TOKENIZERS_ENCODE_METHOD,
    is_local_tokenizer_reference,
    load_token_counter,
)
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


_PROCESS_ORIGIN: float | None = None


@dataclass(frozen=True)
class _RequestSpec:
    index: int
    prompt: str
    scheduled_offset_ms: float


@dataclass(frozen=True)
class _ClientConfig:
    base_url: str
    model: str
    backend: str
    timeout_seconds: float
    api_kind: str
    output_tokens: int
    stream: bool
    concurrency: int
    tokenizer: str | None
    tokenizer_revision: str | None


@dataclass(frozen=True)
class _MeasuredRequest:
    metrics: RequestMetrics
    output_text: str


def _count_prompt_tokens(prompt: str, config: _ClientConfig) -> int:
    if config.backend == "vllm" and config.tokenizer and config.tokenizer_revision:
        return load_token_counter(config.tokenizer, config.tokenizer_revision).count(prompt)
    return estimate_prompt_tokens(prompt)


def _token_count_method(config: _ClientConfig) -> str:
    if config.backend == "vllm":
        return f"prompt=server_usage;output={TOKENIZERS_ENCODE_METHOD}"
    return "mock_synthetic" if config.backend == "mock" else "whitespace_estimate"


def _sleep_until(target: float) -> None:
    while True:
        remaining = target - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.01))


def _execute_request(
    spec: _RequestSpec,
    *,
    origin: float,
    config: _ClientConfig,
    scheduled_offset_ms: float | None = None,
    dispatch_offsets: dict[int, float] | None = None,
) -> _MeasuredRequest:
    scheduled = spec.scheduled_offset_ms if scheduled_offset_ms is None else scheduled_offset_ms
    dispatch = max((time.perf_counter() - origin) * 1000, 0.0)
    if dispatch_offsets is not None:
        dispatch_offsets[spec.index] = dispatch
    client = OpenAICompatibleClient(
        base_url=config.base_url,
        model=config.model,
        backend=config.backend,
        request_timeout_seconds=config.timeout_seconds,
        api_kind=config.api_kind,
        tokenizer=config.tokenizer,
        tokenizer_revision=config.tokenizer_revision,
        defer_token_count=config.backend == "vllm",
    )
    try:
        result = client.complete(spec.prompt, config.output_tokens, spec.index, stream=config.stream)
        completed = max(dispatch + result.total_latency_ms, dispatch)
        metrics = RequestMetrics(
            request_id=f"req-{spec.index + 1:06d}",
            model=config.model,
            backend=config.backend,
            input_tokens=(
                result.input_tokens
                if result.input_tokens is not None
                else _count_prompt_tokens(spec.prompt, config)
            ),
            output_tokens=result.output_tokens,
            concurrency=config.concurrency,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            total_latency_ms=result.total_latency_ms,
            token_count_method=result.token_count_method,
            completed=True,
            error=None,
            started_offset_ms=dispatch,
            completed_offset_ms=completed,
            scheduled_offset_ms=scheduled,
            dispatch_offset_ms=dispatch,
            queue_delay_ms=max(dispatch - scheduled, 0.0),
            end_to_end_latency_ms=max(completed - scheduled, 0.0),
        )
        return _MeasuredRequest(metrics=metrics, output_text=result.output_text)
    except Exception as exc:
        completed = max((time.perf_counter() - origin) * 1000, dispatch)
        metrics = RequestMetrics(
            request_id=f"req-{spec.index + 1:06d}",
            model=config.model,
            backend=config.backend,
            input_tokens=(0 if config.backend == "vllm" else _count_prompt_tokens(spec.prompt, config)),
            output_tokens=0,
            concurrency=config.concurrency,
            ttft_ms=0.0,
            tpot_ms=0.0,
            total_latency_ms=max(completed - dispatch, 0.0),
            token_count_method=_token_count_method(config),
            completed=False,
            error=str(exc),
            started_offset_ms=dispatch,
            completed_offset_ms=completed,
            scheduled_offset_ms=scheduled,
            dispatch_offset_ms=dispatch,
            queue_delay_ms=max(dispatch - scheduled, 0.0),
            end_to_end_latency_ms=max(completed - scheduled, 0.0),
        )
        return _MeasuredRequest(metrics=metrics, output_text="")


def _client_failure_record(
    spec: _RequestSpec,
    *,
    origin: float,
    config: _ClientConfig,
    scheduled_offset_ms: float | None = None,
    error: str,
) -> _MeasuredRequest:
    scheduled = spec.scheduled_offset_ms if scheduled_offset_ms is None else scheduled_offset_ms
    now_offset = max((time.perf_counter() - origin) * 1000, scheduled)
    dispatch = now_offset
    completed = max(now_offset, dispatch)
    metrics = RequestMetrics(
        request_id=f"req-{spec.index + 1:06d}",
        model=config.model,
        backend=config.backend,
        input_tokens=(0 if config.backend == "vllm" else _count_prompt_tokens(spec.prompt, config)),
        output_tokens=0,
        concurrency=config.concurrency,
        ttft_ms=0.0,
        tpot_ms=0.0,
        total_latency_ms=max(completed - dispatch, 0.0),
        token_count_method=_token_count_method(config),
        completed=False,
        error=error,
        started_offset_ms=dispatch,
        completed_offset_ms=completed,
        scheduled_offset_ms=scheduled,
        dispatch_offset_ms=dispatch,
        queue_delay_ms=max(dispatch - scheduled, 0.0),
        end_to_end_latency_ms=max(completed - scheduled, 0.0),
    )
    return _MeasuredRequest(metrics=metrics, output_text="")


def _run_closed_loop_worker(
    requests: list[_RequestSpec],
    *,
    origin: float,
    config: _ClientConfig,
) -> list[_MeasuredRequest]:
    records: list[_MeasuredRequest] = []
    _sleep_until(origin)
    for spec in requests:
        scheduled = max((time.perf_counter() - origin) * 1000, 0.0)
        records.append(
            _execute_request(
                spec,
                origin=origin,
                config=config,
                scheduled_offset_ms=scheduled,
            )
        )
    return records


def _run_request_group(
    requests: list[_RequestSpec],
    *,
    worker_count: int,
    request_schedule: str,
    origin: float | None,
    config: _ClientConfig,
) -> list[_MeasuredRequest]:
    if origin is None:
        origin = _PROCESS_ORIGIN
    if origin is None:
        raise RuntimeError("multiprocess measurement origin was not initialized")
    if not requests:
        return []
    executor = ThreadPoolExecutor(max_workers=worker_count)
    records: list[_MeasuredRequest] = []
    if request_schedule == "closed-loop":
        buckets = [[] for _ in range(worker_count)]
        for index, spec in enumerate(requests):
            buckets[index % worker_count].append(spec)
        closed_loop_futures = [
            executor.submit(_run_closed_loop_worker, bucket, origin=origin, config=config)
            for bucket in buckets
            if bucket
        ]
        for closed_loop_future in as_completed(closed_loop_futures):
            records.extend(closed_loop_future.result())
    else:
        request_futures = []
        for spec in sorted(requests, key=lambda item: item.scheduled_offset_ms):
            _sleep_until(origin + spec.scheduled_offset_ms / 1000)
            request_futures.append(
                executor.submit(
                    _execute_request,
                    spec,
                    origin=origin,
                    config=config,
                )
            )
        for request_future in as_completed(request_futures):
            records.append(request_future.result())
    executor.shutdown(wait=True, cancel_futures=False)
    return records


def _initialize_process(
    ready_queue: object,
    start_event: object,
    origin_value: object,
) -> None:
    global _PROCESS_ORIGIN
    ready_queue.put(os.getpid())
    start_event.wait()
    _PROCESS_ORIGIN = float(origin_value.value)


def _run_measured_requests(
    *,
    prompts: list[str],
    request_schedule: str,
    request_rate_rps: float | None,
    client_processes: int,
    config: _ClientConfig,
) -> list[_MeasuredRequest]:
    interval_ms = 1000 / request_rate_rps if request_schedule == "open-loop" and request_rate_rps else 0.0
    requests = [
        _RequestSpec(index=index, prompt=prompt, scheduled_offset_ms=index * interval_ms)
        for index, prompt in enumerate(prompts)
    ]
    process_groups, worker_counts = _partition_requests(
        requests,
        concurrency=config.concurrency,
        client_processes=client_processes,
    )

    if client_processes == 1:
        origin = time.perf_counter()
        return _run_request_group(
            process_groups[0],
            worker_count=worker_counts[0],
            request_schedule=request_schedule,
            origin=origin,
            config=config,
        )

    records: list[_MeasuredRequest] = []
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    origin_value = context.Value("d", 0.0)
    executor = ProcessPoolExecutor(
        max_workers=client_processes,
        mp_context=context,
        initializer=_initialize_process,
        initargs=(ready_queue, start_event, origin_value),
    )
    future_groups = {
        executor.submit(
            _run_request_group,
            group,
            worker_count=worker_count,
            request_schedule=request_schedule,
            origin=None,
            config=config,
        ): group
        for group, worker_count in zip(process_groups, worker_counts)
    }
    try:
        for _ in range(client_processes):
            ready_queue.get(timeout=30.0)
        origin_value.value = time.perf_counter() + 0.05
        start_event.set()
        for future, group in future_groups.items():
            try:
                records.extend(future.result())
            except Exception as exc:
                for spec in group:
                    records.append(
                        _client_failure_record(
                            spec,
                            origin=float(origin_value.value),
                            config=config,
                            error=f"client process failed: {exc}",
                        )
                    )
    except queue.Empty as exc:
        raise RuntimeError("client processes did not become ready within 30 seconds") from exc
    finally:
        start_event.set()
        executor.shutdown(wait=True, cancel_futures=False)
    return records


def _finalize_token_counts(
    measured: list[_MeasuredRequest],
    config: _ClientConfig,
) -> list[RequestMetrics]:
    if config.backend != "vllm":
        return [item.metrics for item in measured]
    if config.tokenizer is None or config.tokenizer_revision is None:
        raise ValueError("vLLM benchmarks require tokenizer and tokenizer_revision")
    counter = load_token_counter(config.tokenizer, config.tokenizer_revision)
    method = f"prompt=server_usage;output={counter.method}"
    records: list[RequestMetrics] = []
    for item in measured:
        metrics = item.metrics
        if metrics.completed:
            output_tokens = counter.count(item.output_text)
            tpot_ms = 0.0
            if output_tokens > 1:
                tpot_ms = max(metrics.total_latency_ms - metrics.ttft_ms, 0.0) / (
                    output_tokens - 1
                )
            metrics = replace(
                metrics,
                output_tokens=output_tokens,
                tpot_ms=tpot_ms,
                token_count_method=method,
            )
        else:
            metrics = replace(metrics, token_count_method=method)
        records.append(metrics)
    return records


def _partition_requests(
    requests: list[_RequestSpec],
    *,
    concurrency: int,
    client_processes: int,
) -> tuple[list[list[_RequestSpec]], list[int]]:
    base_workers, extra_workers = divmod(concurrency, client_processes)
    worker_counts = [base_workers + (1 if index < extra_workers else 0) for index in range(client_processes)]
    worker_to_process: list[int] = []
    for process_index, worker_count in enumerate(worker_counts):
        worker_to_process.extend([process_index] * worker_count)
    process_groups = [[] for _ in range(client_processes)]
    for spec in requests:
        process_index = worker_to_process[spec.index % concurrency]
        process_groups[process_index].append(spec)
    return process_groups, worker_counts


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
    model_revision: str | None = None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    optimization_profile: str = "baseline",
    server_command_sha256: str | None = None,
    server_command_file: str | Path | None = None,
    request_schedule: str = "closed-loop",
    request_rate_rps: float | None = None,
    client_processes: int = 1,
    queue_delay_warning_ms: float = 10.0,
) -> dict[str, object]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if prompt_texts is not None and not prompt_texts:
        raise ValueError("prompt_texts must not be empty")
    if request_schedule not in {"closed-loop", "open-loop"}:
        raise ValueError("request_schedule must be 'closed-loop' or 'open-loop'")
    if request_schedule == "open-loop" and (
        request_rate_rps is None or not math.isfinite(request_rate_rps) or request_rate_rps <= 0
    ):
        raise ValueError("request_rate_rps must be positive for open-loop scheduling")
    if request_rate_rps is not None and (not math.isfinite(request_rate_rps) or request_rate_rps <= 0):
        raise ValueError("request_rate_rps must be positive")
    if client_processes <= 0:
        raise ValueError("client_processes must be positive")
    if client_processes > concurrency:
        raise ValueError("client_processes must not exceed concurrency")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not math.isfinite(queue_delay_warning_ms) or queue_delay_warning_ms < 0:
        raise ValueError("queue_delay_warning_ms must be non-negative")
    if backend == "vllm" and tokenizer is not None and is_local_tokenizer_reference(tokenizer):
        raise ValueError(
            "local tokenizer paths are mutable and cannot be used for vLLM hardware evidence"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if server_command_file is not None:
        source_command = Path(server_command_file).expanduser().resolve()
        command_bytes = source_command.read_bytes()
        computed_sha256 = hashlib.sha256(command_bytes).hexdigest()
        if server_command_sha256 is not None and computed_sha256 != server_command_sha256:
            raise ValueError("server command file does not match server_command_sha256")
        server_command_sha256 = computed_sha256
        destination = out_dir.resolve() / "server_command.txt"
        write_bytes_atomic(destination, command_bytes)
    memory_before = sample_gpu_memory()
    workload_mode = "fixed_prompts" if prompt_texts is not None else "synthetic"
    workload_fingerprint = prompt_fingerprint(prompt_texts) if prompt_texts is not None else None
    prompt_count = len(prompt_texts) if prompt_texts is not None else None
    shared_tokens = shared_prefix_tokens(prompt_texts) if prompt_texts is not None else None
    shared_fingerprint = shared_prefix_fingerprint(prompt_texts) if prompt_texts is not None else None

    if warmup_count:
        client = OpenAICompatibleClient(
            base_url=base_url,
            model=model,
            backend=backend,
            request_timeout_seconds=timeout_seconds,
            api_kind=api_kind,
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
        )
        warmup_prompts = fixed_prompt_batch(prompt_texts, warmup_count) if prompt_texts is not None else prompt_batch(warmup_count, input_tokens, seed)
        for index, prompt in enumerate(warmup_prompts):
            client.complete(prompt, output_tokens, index, stream=stream)

    records: list[RequestMetrics] = []
    prompts = fixed_prompt_batch(prompt_texts, request_count) if prompt_texts is not None else prompt_batch(request_count, input_tokens, seed + warmup_count)
    if prompt_texts is None:
        workload_fingerprint = prompt_fingerprint(prompts)
    effective_backend = "mock" if base_url.startswith("mock://") else backend
    if effective_backend == "vllm":
        if tokenizer is None or tokenizer_revision is None:
            raise ValueError("vLLM benchmarks require tokenizer and tokenizer_revision")
        token_counter = load_token_counter(tokenizer, tokenizer_revision)
        prompt_token_counts: list[int] = []
        token_count_method = f"prompt=server_usage;output={token_counter.method}"
    else:
        prompt_token_counts = [estimate_prompt_tokens(prompt) for prompt in prompts]
        token_count_method = "mock_synthetic" if effective_backend == "mock" else "whitespace_estimate"
    backend_version = detect_backend_version(effective_backend)
    client_configuration = {
        "request_schedule": request_schedule,
        "request_rate_rps": request_rate_rps,
        "client_processes": client_processes,
        "client_workers": concurrency,
        "queue_delay_warning_ms": queue_delay_warning_ms,
        "timeout_seconds": timeout_seconds,
        "api_kind": api_kind,
        "stream": stream,
    }
    client_config = _ClientConfig(
        base_url=base_url,
        model=model,
        backend=effective_backend,
        timeout_seconds=timeout_seconds,
        api_kind=api_kind,
        output_tokens=output_tokens,
        stream=stream,
        concurrency=concurrency,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
    )
    measured = _run_measured_requests(
        prompts=prompts,
        request_schedule=request_schedule,
        request_rate_rps=request_rate_rps,
        client_processes=client_processes,
        config=client_config,
    )
    records = _finalize_token_counts(measured, client_config)

    records.sort(key=lambda record: record.request_id)
    completed_input_tokens = [record.input_tokens for record in records if record.completed]
    if completed_input_tokens:
        metadata_input_tokens = round(
            sum(completed_input_tokens) / len(completed_input_tokens)
        )
    else:
        metadata_input_tokens = (
            round(sum(prompt_token_counts) / len(prompt_token_counts))
            if prompt_token_counts
            else input_tokens
        )
    measured_elapsed_seconds = measured_span_seconds(records)
    memory_after = sample_gpu_memory()
    resolved_config = {
        "base_url": displayed_base_url(base_url),
        "endpoint_sha256": endpoint_sha256(base_url),
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
        "model_revision": model_revision,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "optimization_profile": optimization_profile,
        "server_command_sha256": server_command_sha256,
        "request_schedule": request_schedule,
        "request_rate_rps": request_rate_rps,
        "client_processes": client_processes,
        "client_workers": concurrency,
        "queue_delay_warning_ms": queue_delay_warning_ms,
        "client_configuration": client_configuration,
        "token_count_method": token_count_method,
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
        base_url=displayed_base_url(base_url),
        endpoint_sha256=endpoint_sha256(base_url),
        api_kind=api_kind,
        dtype=dtype,
        quantization=quantization,
        concurrency=concurrency,
        input_tokens=metadata_input_tokens,
        requested_input_tokens=input_tokens,
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
        model_revision=model_revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        optimization_profile=optimization_profile,
        gpu_driver_version=environment["gpu_driver_version"] if isinstance(environment["gpu_driver_version"], str) else None,
        cuda_version=environment["cuda_version"] if isinstance(environment["cuda_version"], str) else None,
        cuda_driver_api_version=(
            environment["cuda_driver_api_version"]
            if isinstance(environment["cuda_driver_api_version"], str)
            else None
        ),
        torch_version=environment["torch_version"] if isinstance(environment["torch_version"], str) else None,
        server_command_sha256=server_command_sha256,
        stream=stream,
        request_schedule=request_schedule,
        request_rate_rps=request_rate_rps,
        client_processes=client_processes,
        client_workers=concurrency,
        queue_delay_warning_ms=queue_delay_warning_ms,
        client_configuration=client_configuration,
        token_count_method=token_count_method,
    )
    metadata_dict = metadata.to_dict()
    metadata_dict["environment_fingerprint"] = environment_fingerprint(metadata_dict)
    metrics = summarize_requests(records, elapsed_seconds=measured_elapsed_seconds)
    memory = summarize_memory(memory_before, memory_after)
    warnings = _build_run_warnings(
        backend=effective_backend,
        backend_version=backend_version,
        stream=stream,
        memory=memory,
        metrics=metrics,
        request_schedule=request_schedule,
        client_processes=client_processes,
        client_workers=concurrency,
        queue_delay_warning_ms=queue_delay_warning_ms,
    )
    summary = {
        "schema_version": metadata.schema_version,
        "metadata": metadata_dict,
        "metrics": metrics,
        "memory": memory,
        "warnings": warnings,
    }
    write_jsonl(out_dir / "raw_requests.jsonl", records)
    write_request_csv(out_dir / "raw_requests.csv", records)
    write_json(out_dir / "resolved_config.json", resolved_config)
    write_json(out_dir / "run_metadata.json", metadata_dict)
    write_json(out_dir / "summary.json", summary)
    write_summary_markdown(out_dir / "summary.md", summary)
    write_latency_svg(out_dir / "plots" / "latency.svg", records)
    artifacts = [
        "manifest.json",
        "resolved_config.json",
        "raw_requests.jsonl",
        "raw_requests.csv",
        "run_metadata.json",
        "summary.json",
        "summary.md",
        "plots/latency.svg",
    ]
    if (out_dir / "server_command.txt").exists():
        artifacts.append("server_command.txt")
    write_run_manifest(
        out_dir,
        run_type="latency_benchmark",
        artifacts=artifacts,
    )
    return summary


def measured_span_seconds(records: list[RequestMetrics]) -> float:
    if not records:
        return 0.0
    started = min(record.scheduled_offset_ms for record in records)
    completed = max(record.completed_offset_ms for record in records)
    return max(completed - started, 0.0) / 1000


def _build_run_warnings(
    *,
    backend: str,
    backend_version: str | None,
    stream: bool,
    memory: dict[str, object],
    metrics: dict[str, object],
    request_schedule: str,
    client_processes: int,
    client_workers: int,
    queue_delay_warning_ms: float,
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
    if request_schedule == "closed-loop":
        warnings.append(
            "Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions."
        )
    queue_delay = metrics.get("queue_delay_ms", {})
    queue_delay_p95 = float(queue_delay.get("p95", 0.0)) if isinstance(queue_delay, dict) else 0.0
    if queue_delay_p95 > queue_delay_warning_ms:
        warnings.append(
            f"Client saturation detected: queue delay p95 {queue_delay_p95:.3f} ms exceeds the "
            f"{queue_delay_warning_ms:.3f} ms warning threshold."
        )
    if client_processes == 1 and client_workers > 32:
        warnings.append(
            "High concurrency is using a single client process; verify queue delay before interpreting server performance."
        )
    return warnings
