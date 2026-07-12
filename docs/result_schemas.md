# Result Schemas

Result artifacts are part of the project surface. They should remain easy to inspect and version over time.

## Per-Run Directory

Single benchmark commands create timestamped directories under `results/runs/` when no explicit `--output-dir` is supplied.

```text
manifest.json
resolved_config.json
raw_requests.jsonl
raw_requests.csv
run_metadata.json
summary.json
summary.md
plots/latency.svg
```

Throughput benchmark directories include the same raw request and summary artifacts, plus:

```text
throughput_summary.json
throughput_summary.md
```

## `manifest.json`

Every generated run directory includes a manifest with:

- `project_version`
- `schema_version`
- `run_type`
- `artifacts`

The manifest is the first file to inspect when deciding what a run produced.

## `resolved_config.json`

Benchmark and sweep runs persist the resolved configuration used to create the run. Secrets and non-local endpoint URLs are redacted. `api_key_env` may be preserved because it names an environment variable rather than containing the secret value.

## `raw_requests.jsonl`

Each line contains one request record. `raw_requests.csv` contains the same fields in a spreadsheet-friendly table:

- `request_id`
- `model`
- `backend`
- `input_tokens`
- `output_tokens`
- `concurrency`
- `ttft_ms`
- `tpot_ms`
- `total_latency_ms`
- `completed`
- `error`
- `started_offset_ms`
- `completed_offset_ms`

Timeouts and endpoint errors are represented as rows with `completed: false`, zero output tokens, and a populated `error` field.
The request offsets anchor each row to the measured window so wall-clock throughput can be reconstructed from raw evidence rather than trusted from a derived summary.

## `summary.json`

Top-level fields:

- `schema_version`
- `metadata`
- `metrics`
- `memory`
- `warnings`

The `metrics.throughput.measured_elapsed_seconds` field is populated by the benchmark runner's measured wall-clock window. This value is used for output tokens/sec and requests/sec.

The `metrics.timeout_count` field counts failed request rows whose error indicates a timeout. Timeout rows remain visible in `raw_requests.jsonl`; they are not dropped from request counts.

The `metadata` block includes reproducibility fields: project version, Python version, operating system, git commit when available, backend version when available, hardware label, and GPU name when available.

Hardware-oriented metadata also includes `model_revision`, `optimization_profile`, `server_command_sha256`, streaming mode, `gpu_driver_version`, the PyTorch CUDA build as `cuda_version`, the NVIDIA driver API level as `cuda_driver_api_version`, and `torch_version`.
Missing optional hardware fields remain `null` in local or mock runs rather than being inferred.

The `metadata.api_kind` field records whether the run used an OpenAI-compatible `chat` endpoint or `completion` endpoint.

Fixed-prompt benchmark runs add `metadata.workload_mode`, `metadata.prompt_count`, and `metadata.workload_fingerprint`. The fingerprint is a short hash of the prompt set used for comparison safety; prompt text is not stored in the metadata.

Prefix-reuse prompt sets also record `metadata.shared_prefix_tokens_estimate` and `metadata.shared_prefix_fingerprint`. These fields help identify workloads designed for prefix-cache benchmarking without writing shared prompt text to artifacts.

Memory telemetry uses `nvidia-smi` when available. If no NVIDIA GPU telemetry is available, the memory block remains present with `available: false`.

The `warnings` list records unsupported or missing measurements for that run, such as mock-backend limitations, unavailable backend version, unavailable GPU memory telemetry, non-streaming TTFT limitations, or failed requests.

## `throughput_summary.json`

Throughput benchmarks extract a smaller automation-friendly summary from `summary.json`:

- `schema_version`
- `metadata`
- `throughput`
- `request_count`
- `completed_count`
- `failed_count`
- `timeout_count`
- `warnings`

`throughput_summary.md` is the matching human-readable report. Raw per-request evidence remains in `raw_requests.jsonl` and `raw_requests.csv`.

## KV Cache Estimate

`llm-accel kv-cache estimate --json` writes a single JSON object with:

- `preset`, when a built-in model-shape preset was used
- `layers`
- `sequence_length`
- `batch_size`
- `kv_heads`
- `head_dim`
- `dtype`
- `dtype_bytes`
- `bytes`
- `mib`
- `gib`
- `explanation`

## Sweep Aggregate

Sweep runs write:

```text
aggregate_summary.json
aggregate_summary.md
plots/sweep_throughput.svg
plots/latency_throughput.svg
```

## Quality Evaluation

`llm-accel eval sanity` writes:

```text
manifest.json
quality_eval.json
quality_eval.md
```

`llm-accel eval task` writes:

```text
manifest.json
task_eval.json
task_eval.md
```

## Validation and Comparison

```bash
llm-accel report generate --run-dir results/runs/example
llm-accel report validate --run-dir results/runs/example
llm-accel report compare --summary run-a/summary.json --summary run-b/summary.json --output-dir results/runs/compare
llm-accel report claim-audit --run-dir results/runs/example
```

`report generate --run-dir` is analysis-only: it reads existing artifacts and regenerates `summary.md` plus `plots/latency.svg` when raw request records are available.

`report validate --run-dir` checks the manifest artifact list and validates the key fields for benchmark summaries, throughput summaries, sweep aggregates, comparison reports, evaluation reports, vLLM validation reports, vLLM benchmark plans, quantization comparisons, and speculative-decoding reports.

Comparison runs write:

```text
manifest.json
comparison.json
comparison.md
```

`comparison.json` includes `comparable`, `ranking_allowed`, and `warnings`.
Reports must treat relative throughput as an inspection aid rather than a ranking when runs are too small, have failures, have fewer than three repetitions per profile, or differ in model revision, backend version, API and streaming mode, dtype, quantization, hardware and accelerator software, concurrency, request count, or workload shape.

Repeated-run comparisons include `profile_aggregates` with repetition count plus mean, population standard deviation, minimum, and maximum throughput and p95 latency for each optimization profile.

## vLLM Validation

`llm-accel vllm validate` writes:

```text
manifest.json
vllm_validation.json
vllm_validation.md
```

`llm-accel vllm plan` writes:

```text
manifest.json
vllm_benchmark_plan.json
vllm_benchmark_plan.md
server_command.txt
```

## Speculative Decoding

`llm-accel speculative run` writes:

```text
manifest.json
speculative_summary.json
speculative_summary.md
acceptance_curve.json
baseline_comparison.json
baseline_comparison.md
```

## Quantization Comparison

`quantization_comparison.json` records:

- requested `modes`
- backend `supported_modes`
- per-mode `support_status`
- whether the mode was `measured`
- per-mode summary path and quality sanity result when measured
- `warnings` for unsupported or unknown support
