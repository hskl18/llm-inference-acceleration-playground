# Command Reference

Every command below runs against the deterministic `mock://local` backend unless it names a real endpoint, so the whole walkthrough works without a GPU.
Mock output proves the workflow and the schemas; it is never a performance claim.

See [docs/features.md](features.md) for the full feature inventory and [docs/result_schemas.md](result_schemas.md) for what each artifact contains.

## Example configs

```bash
llm-accel examples list
llm-accel examples write --output-dir configs
```

## Benchmarks

Run a local smoke benchmark:

```bash
llm-accel bench latency \
  --base-url mock://local \
  --model mock-model \
  --api-kind chat \
  --concurrency 4 \
  --input-tokens 128 \
  --output-tokens 64 \
  --request-count 8 \
  --hardware-label local-dev \
  --output-dir results/runs/readme-smoke
```

If `--output-dir` is omitted, benchmark commands create a timestamped directory under `results/runs/`.

Generated files:

- `manifest.json`
- `raw_requests.jsonl`
- `raw_requests.csv`
- `resolved_config.json`
- `run_metadata.json`
- `summary.json`
- `summary.md`
- `plots/latency.svg`

Use `--api-kind completion` for OpenAI-compatible `/v1/completions` endpoints instead of chat-completion endpoints.

Run a fixed-prompt benchmark without storing prompt text in result metadata:

```bash
llm-accel bench latency \
  --base-url mock://local \
  --model mock-model \
  --prompts configs/spec_prompts.jsonl \
  --request-count 4 \
  --output-dir results/runs/readme-prompts
```

Run a throughput-focused benchmark:

```bash
llm-accel bench throughput \
  --base-url mock://local \
  --model mock-model \
  --concurrency 4 \
  --output-tokens 64 \
  --request-count 8 \
  --output-dir results/runs/readme-throughput
```

Throughput runs preserve the standard raw request artifacts and add `throughput_summary.json` plus `throughput_summary.md`.

## Goodput and repeated runs

Declare a latency budget and the summary reports how much of the load actually met it:

```bash
llm-accel bench throughput \
  --base-url mock://local \
  --request-count 32 \
  --slo-ttft-ms 1000 \
  --slo-tpot-ms 15 \
  --output-dir results/runs/goodput-example
```

Repeat the whole run to separate a real difference from run-to-run noise:

```bash
llm-accel bench throughput \
  --base-url mock://local \
  --request-count 32 \
  --repeats 3 \
  --output-dir results/runs/repeats-example
```

Each repetition lands in `repeat-01` through `repeat-NN`, and `repeats_summary.json` reports every metric with a 95% confidence interval for its mean.

When the repetitions cannot rerun one identical configuration, for example because the server caches prompts and each repetition needs a fresh prompt set, run them separately and aggregate afterwards:

```bash
llm-accel report repeats \
  --run-dir results/runs/study/repeat-01 \
  --run-dir results/runs/study/repeat-02 \
  --run-dir results/runs/study/repeat-03 \
  --output-dir results/runs/study
```

## Arrival scheduling

Use open-loop arrivals to expose client backlog under offered load:

```bash
llm-accel bench throughput \
  --base-url http://localhost:8000/v1 \
  --backend vllm \
  --request-schedule open-loop \
  --request-rate-rps 20 \
  --concurrency 8 \
  --client-processes 2 \
  --queue-delay-warning-ms 10 \
  --output-dir results/runs/open-loop-example
```

Closed-loop runs remain useful for bounded-concurrency inspection, but their summaries warn that response-dependent arrivals are susceptible to coordinated omission.

## Sweeps and matrices

```bash
llm-accel bench sweep --config configs/benchmark_small.yaml
llm-accel bench sweep --config configs/benchmark_prompts.yaml
llm-accel bench sweep --config configs/benchmark_prefix_cache.yaml
```

```bash
llm-accel bench matrix \
  --config configs/optimization_matrix_mock.yaml \
  --output-dir results/runs/mock-optimization-matrix
```

The matrix covers baseline, prefix cache, chunked prefill, quantized, and speculative treatment profiles in randomized order for three repetitions.
It checkpoints `matrix_state.json` after every cell and resumes only when the config digest and existing run artifacts remain valid.
Real matrices require one explicit, distinct, already-running endpoint URL per profile.
The tool does not provision hardware, download models, or silently restart serving processes.

## Reports and audits

```bash
llm-accel report generate --run-dir results/runs/readme-smoke
llm-accel report validate --run-dir results/runs/readme-smoke
llm-accel report claim-audit --run-dir results/runs/readme-smoke
llm-accel report ranking-audit --matrix-dir results/runs/mock-optimization-matrix
llm-accel report compare \
  --summary results/runs/run-a/summary.json \
  --summary results/runs/run-b/summary.json \
  --output-dir results/runs/comparison
```

Comparison reports include structured blockers, invariant strata, and `ranking_allowed`.
Optimization settings are treatment dimensions, while model, tokenizer, prompt, schedule, client, quality-gate, and environment evidence remain comparison invariants.
Relative throughput is computed from the declared baseline aggregate rather than input order.
The claim audit intentionally rejects a mock smoke run because it is not hardware evidence.
The ranking audit also rejects the mock matrix, closed-loop coordinated-omission risk, client saturation, missing repetitions, missing quality deltas, and any corrupted source evidence.

## KV cache

```bash
llm-accel kv-cache estimate \
  --preset llama-3-8b \
  --seq-len 8192 \
  --batch-size 16 \
  --dtype fp16 \
  --json

llm-accel kv-cache presets
```

## Backends and vLLM helpers

```bash
llm-accel doctor
llm-accel backend list
llm-accel backend profile --backend vllm --base-url http://localhost:8000/v1
```

Generate a vLLM OpenAI-compatible server command:

```bash
llm-accel vllm command \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dtype auto \
  --port 8000 \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

The output is a `vllm serve <model>` command line.
Prefix caching and chunked prefill are always stated explicitly, because vLLM enables both by default, and speculative settings are emitted as one `--speculative-config` JSON object.

```bash
llm-accel vllm validate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --base-url http://localhost:8000/v1 \
  --output-dir results/runs/vllm-validation

llm-accel vllm plan \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --hardware-label GPU_CLASS \
  --dtype float16 \
  --base-url http://localhost:8000/v1 \
  --output-dir results/runs/vllm-plan
```

## Quantization comparison

```bash
llm-accel quantization compare \
  --model MODEL_ID \
  --backend vllm \
  --mode none=http://localhost:8000/v1 \
  --mode awq=http://localhost:8001/v1 \
  --output-dir results/runs/quantization-comparison
```

A quantization mode is a property of the loaded weights, so two modes may not share one endpoint.
The first mode is the baseline, and every other mode is scored against it with a temperature-0 exact-match rate plus a perplexity delta where the backend returns prompt logprobs.
Requested modes are labeled as `supported`, `unsupported`, or `unknown`; unsupported modes are reported but not benchmarked.

## Quality evaluation

```bash
llm-accel eval sanity \
  --base-url mock://local \
  --model mock-model \
  --prompts configs/spec_prompts.jsonl \
  --output-dir results/runs/eval-smoke

llm-accel eval task \
  --base-url mock://local \
  --model mock-model \
  --tasks configs/task_eval_small.jsonl \
  --output-dir results/runs/task-eval-smoke
```

## Speculative decoding analysis

```bash
llm-accel speculative run \
  --lookahead 4 \
  --acceptance-rate 0.7 \
  --draft-cost-ratio 0.2 \
  --output-dir results/runs/speculative-analysis
```

This evaluates an analytical model, not a measured speedup.
Pass `--metrics-base-url` to read a real acceptance rate from a vLLM Prometheus endpoint instead of assuming one.
