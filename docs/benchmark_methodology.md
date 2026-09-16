# Benchmark Methodology

Benchmarks should record workload shape, backend, backend version when available, model, dtype, quantization mode, request count, warmup count, software environment, hardware label, GPU name when available, generated metrics, and warnings about missing measurements.

`backend_version` for a vLLM endpoint is read from the server's `GET /version`, because the client is usually not the machine that serves the requests.
`backend_version_source` records where it came from: `server_version_endpoint`, `client_package`, `mock`, or `unavailable`.
A run whose version came from the client's installed package carries a warning, since it only describes the endpoint on a same-host run.

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
Every streaming request asks for `stream_options.include_usage`, so `openai-compatible`, `sglang`, and `tgi` runs use server-reported usage whenever the backend returns it.
`metadata.token_count_method` is derived from what the endpoint actually reported rather than assumed before the run: `server_usage` when usage was returned, `whitespace_estimate` when it was not, and `mixed:` when requests in one run disagreed.
Runs that fall back to a whitespace estimate, or that mix methods, say so in `summary.json` warnings.
- p50, p95, p99 latency
- inter-token latency p50, p95, p99, and maximum
- goodput against declared TTFT, TPOT, and end-to-end SLOs
- failed request count
- timeout count
- client queue delay p50, p95, p99, and maximum
- scheduled-arrival to completion latency

## Inter-Token Latency

TPOT is one number per request: the decode span divided by the output tokens after the first.
It hides stalls, so a run that pauses for 400 ms in the middle of a response can report the same TPOT as a smooth one.

Streaming runs therefore also record the gap between consecutive content-bearing stream chunks on every request, in `raw_requests.jsonl` as `inter_token_latencies_ms`.
`metrics.inter_token_latency_ms` pools those samples across completed requests and reports the count, mean, p50, p95, p99, and maximum.

These are chunk-arrival gaps observed by the client.
They equal per-token latency only when the server emits one token per chunk, which is why they are reported separately from `tpot_ms` rather than replacing it.
Non-streaming runs record no samples at all, and the claim audit blocks a run that has none.

## Goodput

Average latency and average throughput can both look healthy while a large share of requests misses the latency budget the service actually promises.

Declare the budget with `--slo-ttft-ms`, `--slo-tpot-ms`, and `--slo-e2e-ms`.
A request counts toward goodput only when it completed and met every declared threshold.
`metrics.goodput` then reports the declared `slo`, `good_request_count`, `attainment` over all measured requests, `good_requests_per_second`, and `good_output_tokens_per_second`.
Without a declared SLO, `metrics.goodput` is `null`; nothing is assumed on the user's behalf.

The SLO is a reporting threshold rather than a workload property, so it is recorded in `metadata.slo` but kept out of `metadata.client_configuration`.
Two runs that differ only in their declared SLO stay in one comparison stratum.

## Repeated Runs

A single run reports percentiles over requests, which says nothing about how much the result moves when the same configuration is run again.

`--repeats N` runs the whole configuration N times into `repeat-01` through `repeat-NN` and writes `repeats_summary.json` and `repeats_summary.md` beside them.
Each aggregated metric carries the repetition count, mean, sample standard deviation, minimum, maximum, and a two-sided 95% Student t confidence interval for the mean.
The interval assumes the repetitions are independent draws whose mean is approximately normal, and back-to-back repetitions on one host capture run-to-run noise rather than day-to-day drift.
A metric that any repetition failed to report is left out instead of being averaged over a partial set.

## Client Bottleneck Detection

A load generator that cannot keep up reports its own limits as server latency, so every run publishes two independent client signals.

`metrics.queue_delay_ms` measures backlog: how long a scheduled arrival waited before a client worker dispatched it.
A queue-delay p95 above `--queue-delay-warning-ms` raises a client-saturation warning, and the ranking audit blocks a saturated run outright.

`metrics.client_load` measures cost: the CPU seconds the load generator itself burned over the measured span, expressed as `cpu_cores_used`.
Above 0.8 cores a single-process client is near the limit of one core under the interpreter lock, and the run warns to add `--client-processes` before attributing latency to the server.
`time.process_time` does not count spawned client processes, so multiprocess runs set `covers_all_client_processes` to `false` and raise no CPU warning; their backlog signal remains queue delay.

Hardware-backed runs also record model revision, optimization profile, GPU driver, CUDA, PyTorch, backend version, repository commit, and GPU memory when the host exposes them.
Use `llm-accel report claim-audit` before treating a run as publishable hardware evidence.
The audit is a minimum evidence gate, not a substitute for repeated runs, compatible comparisons, or quality evaluation.

## Streaming and Non-Streaming Timing

Streaming endpoint calls observe TTFT from the first server-sent event that carries non-empty generated text.
Role-only chunks, `null` or empty `content`, and usage-only chunks with `choices: []` do not start TTFT and never add text.
Reasoning deltas (`reasoning_content`, or `reasoning` in newer vLLM releases) are generated tokens, so they start TTFT and count toward output tokens.
They are kept out of the recorded output text so quality validators only see the final answer.
Non-streaming endpoint calls cannot observe first-token timing, so TTFT is conservatively recorded as total request latency.
A non-streaming `message.content` of `null` is recorded as empty output text.

## Output Length

Temperature-0 generations can stop at an EOS or stop token before `--output-tokens`, which makes token throughput incomparable across configurations.
Benchmark requests therefore send `ignore_eos` together with `min_tokens` equal to the requested output length, so every request generates exactly that many tokens.
`min_tokens` is sent as well because `ignore_eos` alone only skips the tokenizer EOS token, not other stop token ids.
This is on by default for vLLM runs and off for every other backend; enable or disable it with `--ignore-eos` and `--no-ignore-eos`, or with `workload.ignore_eos` in a sweep or matrix config.
The resolved value is recorded as `metadata.ignore_eos` and inside `metadata.client_configuration`, which is a comparison invariant, so runs with and without a fixed output length are not pooled into one comparison.
A run whose completed requests returned different output token counts carries a warning, and the claim audit warns when a vLLM run did not fix its output length.

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
Each synthetic prompt is drawn from a seeded PRNG over a fixed vocabulary, so every measured request gets its own prompt and prompts share no prefix beyond chance.
This matters because vLLM enables automatic prefix caching by default, so a repeating or prefix-sharing synthetic workload would silently measure cache hits.
Warmup requests use prompt indices after the measured ones, so a warmup request never pre-populates a cache entry for a measured prompt, and changing the warmup count does not change the measured workload.
Every run records `unique_prompt_count`, and a run whose measured prompts repeat carries a warning in `summary.json`.

Prompt reuse is an explicit choice.
Latency and throughput benchmarks accept `--prompts` with plain-text lines or JSONL records containing a `prompt` field, and config sweeps can use `workload.prompts_path` for the same fixed-prompt behavior.
Use a prompt file when the experiment is about prefix reuse or caching.

Prompt text is sent to the configured endpoint but is not written into result metadata.
Fixed-prompt benchmark metadata records `workload_mode`, `prompt_count`, and a short prompt-set fingerprint so comparisons can detect mismatched prompt sets without exposing prompt contents.

Every run records the measured prompt-set fingerprint, `unique_prompt_count`, an estimated shared-prefix token count, and a shared-prefix fingerprint.
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
`endpoint.api_key_env` names the environment variable the client reads for the bearer token, and a matrix profile may override it because profiles use distinct endpoints.
The CLI equivalent is `--api-key-env`; it defaults to `OPENAI_API_KEY`.
Only the variable name is used and recorded, never the value, and requests are sent without an Authorization header when the variable is unset.

## Optimization Profiles

Every matrix cell writes `optimization_profile.json` using schema `0.2`.
The profile records the backend and exact version, exact server command text and parsed arguments, command SHA-256, target model and immutable revision, tokenizer and immutable revision, dtype, quantization, prefix-cache state, chunked-prefill state, speculative model settings, batching limits, model limits, GPU-memory limit, and environment fingerprint.

For a vLLM profile, the audit matches those fields against the exact command: the model is the positional argument of `vllm serve`, prefix caching and chunked prefill must be stated explicitly because vLLM enables both by default, and the speculative model and token count are matched against the `--speculative-config` JSON object.

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
