# Benchmark Methodology

Benchmarks should record workload shape, backend, backend version when available, model, dtype, quantization mode, request count, warmup count, software environment, hardware label, GPU name when available, generated metrics, and warnings about missing measurements.

The first implementation supports deterministic mock benchmarks for smoke testing. Mock results validate the workflow and schemas; they are not hardware performance claims.

Endpoint benchmarks support both OpenAI-compatible `/v1/chat/completions` and `/v1/completions` APIs. Use `--api-kind chat` for chat-completion servers and `--api-kind completion` for legacy completion-style servers.

Primary metrics:

- TTFT: request start to first token
- TPOT: time per output token after the first token
- total latency: request start to final token
- output tokens/sec
- requests/sec
- p50, p95, p99 latency
- failed request count
- timeout count

Hardware-backed runs also record model revision, optimization profile, GPU driver, CUDA, PyTorch, backend version, repository commit, and GPU memory when the host exposes them.
Use `llm-accel report claim-audit` before treating a run as publishable hardware evidence.
The audit is a minimum evidence gate, not a substitute for repeated runs, compatible comparisons, or quality evaluation.

## Streaming and Non-Streaming Timing

Streaming endpoint calls observe TTFT from the first content-bearing server-sent event. Non-streaming endpoint calls cannot observe first-token timing, so TTFT is conservatively recorded as total request latency.

## Concurrency

The benchmark runner executes measured requests with a worker pool sized by the configured `concurrency`. Summary throughput uses measured wall-clock elapsed time for the measured window, while raw request records retain per-request latency.
Raw rows also record request start and completion offsets from the measured-window origin.
The hardware claim audit rebuilds the measured span and derived throughput from those offsets.

`bench throughput` uses the same request execution path as latency benchmarking so TTFT, latency, failure, and timeout records stay comparable. It adds throughput-focused summary artifacts while preserving raw request evidence.

## Workloads

The default benchmark workload is synthetic and controlled by `--input-tokens`, `--output-tokens`, `--request-count`, and `--seed`. Latency and throughput benchmarks also accept `--prompts` with plain-text lines or JSONL records containing a `prompt` field. Config sweeps can use `workload.prompts_path` for the same fixed-prompt behavior.

Prompt text is sent to the configured endpoint but is not written into result metadata. Fixed-prompt benchmark metadata records `workload_mode`, `prompt_count`, and a short prompt-set fingerprint so comparisons can detect mismatched prompt sets without exposing prompt contents.

For prefix-reuse workloads, metadata also records an estimated shared-prefix token count and a shared-prefix fingerprint. Use `configs/benchmark_prefix_cache.yaml` as a small workflow check for prefix-cache experiments before moving to a real long-document workload.

## Run Directories

When `--output-dir` is omitted, single benchmark commands write to a timestamped directory under `results/runs/`. Explicit output directories are preserved exactly so scripted experiments can choose stable paths.

## Timeouts and Failures

Benchmark timeouts are recorded as failed request rows in `raw_requests.jsonl` and `raw_requests.csv`; they do not discard the rest of the run. Summaries include failed request counts, timeout counts, and error rate.

## Warnings

Each `summary.json` includes a `warnings` list. Warnings are part of the benchmark artifact because missing GPU telemetry, unavailable backend version, non-streaming TTFT limitations, mock backend runs, and failed requests affect how results should be interpreted.

## Config Validation

Sweep configs are validated before any run starts. Required endpoint, model, run, and workload fields must be present; request counts, timeouts, token lengths, and concurrency values must be positive where applicable. Endpoint secrets must be referenced through `api_key_env`, not embedded directly in config files.

## Plots

Each latency benchmark writes `plots/latency.svg`. The plot is intentionally dependency-free and should be treated as a quick inspection artifact; raw JSONL remains the source of truth. `raw_requests.csv` mirrors the JSONL fields for spreadsheet inspection.

Config-defined sweeps also write `aggregate_summary.json`, `aggregate_summary.md`, `plots/sweep_throughput.svg`, and `plots/latency_throughput.svg`.

Raw request records, resolved config, run metadata, and `summary.json` are written before Markdown and plot report artifacts. This preserves machine-readable benchmark evidence when report generation fails.

Throughput benchmark runs also write `throughput_summary.json` and `throughput_summary.md`. These files extract output tokens/sec, requests/sec, measured elapsed time, completed requests, failed requests, and timeout count from the same measured run.

## Validation

Generated run directories can be checked with:

```bash
llm-accel report generate --run-dir results/runs/example
llm-accel report validate --run-dir results/runs/example
llm-accel report claim-audit --run-dir results/runs/example
```

`report generate` regenerates `summary.md` and `plots/latency.svg` from existing `summary.json` and `raw_requests.jsonl` without rerunning inference. The validator checks manifest artifacts, schema version, required summary fields, aggregate run counts, throughput summaries, comparison reports, evaluation reports, vLLM validation reports, vLLM runbooks, quantization comparisons, and speculative-decoding artifacts.

## Comparisons

`llm-accel report compare` emits machine-readable `warnings`, `comparable`, and `ranking_allowed` fields. Relative throughput is not a ranking when runs are too small, contain failures, or differ in model, backend, dtype, quantization, hardware label, or workload shape.
