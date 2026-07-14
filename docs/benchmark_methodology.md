# Benchmark Methodology

Benchmarks should record workload shape, backend, backend version when available, model, dtype, quantization mode, request count, warmup count, software environment, hardware label, GPU name when available, generated metrics, and warnings about missing measurements.

The first implementation supports deterministic mock benchmarks for smoke testing.
Mock results validate the workflow and schemas; they are not hardware performance claims.

Endpoint benchmarks support both OpenAI-compatible `/v1/chat/completions` and `/v1/completions` APIs.
Use `--api-kind chat` for chat-completion servers and `--api-kind completion` for legacy completion-style servers.

Primary metrics:

- TTFT: request start to first token
- TPOT: time per output token after the first token
- total latency: request start to final token
- output tokens/sec
- requests/sec

vLLM runs use the server-reported prompt token count so chat-template and server-added tokens are included.
They join the final generated text and count output tokens with the declared tokenizer at its immutable revision using `encode(add_special_tokens=False)`.
Streaming responses are joined before the final tokenizer count, so token boundaries that span chunks remain correct.
Output tokenization happens after all endpoint measurements finish, so tokenizer execution time cannot inflate latency, delay closed-loop dispatch, or reduce measured throughput.
Mutable local tokenizer paths are excluded from hardware evidence because an immutable revision cannot bind their content.
Whitespace estimates remain explicit for generic compatibility runs and cannot satisfy the vLLM hardware-claim gate.
- p50, p95, p99 latency
- failed request count
- timeout count
- client queue delay p50, p95, p99, and maximum
- scheduled-arrival to completion latency

Hardware-backed runs also record model revision, optimization profile, GPU driver, CUDA, PyTorch, backend version, repository commit, and GPU memory when the host exposes them.
Use `llm-accel report claim-audit` before treating a run as publishable hardware evidence.
The audit is a minimum evidence gate, not a substitute for repeated runs, compatible comparisons, or quality evaluation.

## Streaming and Non-Streaming Timing

Streaming endpoint calls observe TTFT from the first content-bearing server-sent event.
Non-streaming endpoint calls cannot observe first-token timing, so TTFT is conservatively recorded as total request latency.

## Arrival Scheduling and Concurrency

The benchmark runner separates prompt source from request arrival scheduling.
`workload_mode` identifies synthetic or fixed-prompt input, while `request_schedule` identifies closed-loop or open-loop arrivals.

Closed-loop mode maintains the configured number of logical workers.
Each worker sends its next request only after its prior request completes.
This is useful for bounded-concurrency inspection, but request arrivals depend on response time and are therefore susceptible to coordinated omission.
Every closed-loop summary records that warning.

Open-loop mode schedules deterministic fixed-cadence arrivals from `request_rate_rps`.
Each raw row records `scheduled_offset_ms`, `dispatch_offset_ms`, `queue_delay_ms`, `started_offset_ms`, `completed_offset_ms`, and `end_to_end_latency_ms`.
The scheduled offset is the intended arrival time.
The dispatch offset is when a client worker actually begins the endpoint call.
Queue delay is dispatch minus schedule.
End-to-end latency includes client backlog from scheduled arrival through completion.

Summary throughput uses the scheduled measurement origin through the final completion.
The hardware claim audit rebuilds that span and derived metrics from raw rows.
The ranking audit applies the stricter of the configured warning threshold and a hard ceiling derived from the offered request cadence.
If queue-delay p95 exceeds that effective threshold, the run is marked as client-saturated and cannot support a performance ranking.

`client_processes` optionally distributes the global concurrency across spawned processes.
Each process owns its HTTP clients and local thread workers.
The process count must not exceed total concurrency.
Single-process and multiprocess runs are different client configurations and are not silently pooled into one comparison stratum.

`bench throughput` uses the same request execution path as latency benchmarking so TTFT, latency, failure, and timeout records stay comparable.
It adds throughput-focused summary artifacts while preserving raw request evidence.

## Workloads

The default benchmark workload is synthetic and controlled by `--input-tokens`, `--output-tokens`, `--request-count`, and `--seed`.
Latency and throughput benchmarks also accept `--prompts` with plain-text lines or JSONL records containing a `prompt` field.
Config sweeps can use `workload.prompts_path` for the same fixed-prompt behavior.

Prompt text is sent to the configured endpoint but is not written into result metadata.
Fixed-prompt benchmark metadata records `workload_mode`, `prompt_count`, and a short prompt-set fingerprint so comparisons can detect mismatched prompt sets without exposing prompt contents.

For prefix-reuse workloads, metadata also records an estimated shared-prefix token count and a shared-prefix fingerprint.
Use `configs/benchmark_prefix_cache.yaml` as a small workflow check for prefix-cache experiments before moving to a real long-document workload.

## Run Directories

When `--output-dir` is omitted, single benchmark commands write to a timestamped directory under `results/runs/`.
Explicit output directories are preserved exactly so scripted experiments can choose stable paths.

## Timeouts and Failures

Benchmark timeouts are recorded as failed request rows in `raw_requests.jsonl` and `raw_requests.csv`; they do not discard the rest of the run.
Summaries include failed request counts, timeout counts, and error rate.

## Warnings

Each `summary.json` includes a `warnings` list.
Warnings are part of the benchmark artifact because missing GPU telemetry, unavailable backend version, non-streaming TTFT limitations, mock backend runs, and failed requests affect how results should be interpreted.

## Config Validation

Sweep and matrix configs are validated before any run starts.
Required endpoint, model, run, and workload fields must be present.
Request counts, timeouts, token lengths, offered request rate, concurrency, process counts, and queue thresholds are checked where applicable.
Endpoint secrets must be referenced through `api_key_env`, not embedded directly in config files.

## Optimization Profiles

Every matrix cell writes `optimization_profile.json` using schema `0.2`.
The profile records the backend and exact version, exact server command text and parsed arguments, command SHA-256, target model and immutable revision, tokenizer and immutable revision, dtype, quantization, prefix-cache state, chunked-prefill state, speculative model settings, batching limits, model limits, GPU-memory limit, and environment fingerprint.

The semantic fingerprint covers the complete profile except its display name.
The treatment fingerprint covers settings that intentionally differ between experiment arms.
The exact command byte hash remains an explicit field and participates in profile identity, so even whitespace changes remain visible rather than being normalized away.
The summary and raw request rows also bind the token-count method used for TPOT and output tokens/sec.

Matrix profiles are experimental treatments.
Model, tokenizer, workload, arrival schedule, client configuration, quality gate, environment, request shape, warmups, and request counts are comparison invariants.
Missing invariant evidence does not compare equal merely because it is missing in every run.

## Randomized Matrices

`llm-accel bench matrix` requires baseline, prefix-cache, chunked-prefill, quantized, and speculative profile definitions plus at least three repetitions.
Profile order is randomized independently within each repetition from the persisted seed.
Warmup requests run before each measured cell and do not appear in raw measured rows.

`matrix_plan.json` fixes the complete randomized plan before execution.
`matrix_state.json` checkpoints pending, running, successful, evidence-failed, and execution-failed cells after every transition.
Resume rejects a changed config digest and skips only existing cells whose manifests and summaries still validate.

Real profiles must use distinct explicit endpoint URLs because the runner does not provision or restart serving infrastructure.
Mock profiles may share `mock://local` because they validate orchestration only.

## Plots

Each latency benchmark writes `plots/latency.svg`.
The plot is intentionally dependency-free and should be treated as a quick inspection artifact; raw JSONL remains the source of truth.
`raw_requests.csv` mirrors the JSONL fields for spreadsheet inspection.

Config-defined sweeps also write `aggregate_summary.json`, `aggregate_summary.md`, `plots/sweep_throughput.svg`, and `plots/latency_throughput.svg`.

Raw request records, resolved config, run metadata, and `summary.json` are written before Markdown and plot report artifacts.
This preserves machine-readable benchmark evidence when report generation fails.

Throughput benchmark runs also write `throughput_summary.json` and `throughput_summary.md`.
These files extract output tokens/sec, requests/sec, measured elapsed time, completed requests, failed requests, and timeout count from the same measured run.

## Validation

Generated run directories can be checked with:

```bash
llm-accel report generate --run-dir results/runs/example
llm-accel report validate --run-dir results/runs/example
llm-accel report claim-audit --run-dir results/runs/example
llm-accel report ranking-audit --matrix-dir results/runs/optimization-matrix
```

`report generate` regenerates `summary.md` and `plots/latency.svg` from existing `summary.json` and `raw_requests.jsonl` without rerunning inference.
The validator checks manifest artifacts, schema version, required summary fields, aggregate run counts, throughput summaries, comparison reports, evaluation reports, vLLM validation reports, vLLM runbooks, quantization comparisons, and speculative-decoding artifacts.

## Comparisons and Ranking Audit

`llm-accel report compare` emits structured blockers, invariant strata, per-profile valid-repetition aggregates, `comparable`, and `ranking_allowed`.
Strict mode blocks an experiment when invariant fingerprints differ.
Stratified mode may compute separate within-stratum aggregates, but cross-stratum ranking remains blocked.
Quantization and other optimization settings are allowed treatment differences when the shared invariants match.

The declared baseline profile defines relative aggregate throughput.
Randomized input order never selects the denominator.
Failed cells remain visible and do not count toward the three required valid repetitions.

`llm-accel report ranking-audit` follows every source run from the matrix bundle.
It requires raw traces, exact profile and command evidence, hardware telemetry, request-count parity, common quality-suite fingerprints, disclosed quality deltas, three valid repetitions per profile, open-loop dispatch evidence, a non-saturated client, and one compatible comparison stratum.
The audit blocks mock data and records the remaining operator-evidence limitation for already-running server processes.
