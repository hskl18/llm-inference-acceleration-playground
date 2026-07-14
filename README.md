# LLM Inference Acceleration Playground

Open source toolkit for measuring LLM inference-serving latency, throughput, KV cache memory, and acceleration tradeoffs.

The project focuses on practical benchmark workflows around OpenAI-compatible serving endpoints.
The first backend target is vLLM, while the current implementation includes a deterministic `mock://local` path so contributors can run tests and smoke benchmarks without a GPU.

## Evidence Status

No real GPU benchmark is checked into this repository yet.
Current mock runs prove the client, artifact, validation, and reporting workflow only.
They do not support latency, throughput, memory, quality, or acceleration claims about vLLM or any model.

The next hardware experiment is fully specified in [the hardware benchmark runbook](docs/hardware_benchmark_runbook.md).
It collects three or more repetitions for baseline, prefix-cache, chunked-prefill, quantized, and speculative profiles.
It records the exact hardware and software environment and blocks publication when required evidence is absent.

## Current Status

Version 0.2.0 adds measurement-correct experiment orchestration while preserving the deterministic no-GPU workflow:

- Python package scaffold under `src/llm_accel`
- `llm-accel` CLI
- mock latency and throughput benchmark path
- OpenAI-compatible non-streaming and streaming endpoint client
- closed-loop and deterministic open-loop request scheduling
- scheduled-arrival, actual-dispatch, client-queue, and end-to-end request timing
- optional spawned multiprocess load generation
- YAML/JSON sweep config loading
- synthetic and fixed-prompt benchmark workloads
- raw JSONL, summary JSON, and summary Markdown outputs
- SVG latency plots
- KV cache estimator
- backend capability matrix
- vLLM server command helper
- optional GPU memory telemetry through `nvidia-smi`
- quantization comparison workflow
- fixed-prompt quality sanity checks for quantization comparisons
- standalone lightweight quality sanity evaluation
- exact-match, regex, Draft 2020-12 JSON Schema, long-context, and keyword task validators
- separate task-specification, raw-output, and generated-summary quality artifacts
- run manifests for generated artifacts
- run validation and cross-run comparison reports
- versioned structured optimization profiles with exact command and environment fingerprints
- randomized, resumable five-profile matrices with three or more repetitions
- strict and explicitly stratified comparison modes
- single-run hardware claim audits and matrix-level performance-ranking audits
- endpoint health checks through `doctor`
- backend adapter profiles
- packaged example configs for installed CLI users
- toy speculative decoding accounting
- speculative decoding acceptance-curve reports
- unit and integration tests

OpenAI-compatible endpoint calls are implemented with the Python standard library.
Streaming mode records observed TTFT from server-sent events.
Non-streaming mode conservatively records TTFT as total request latency.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Quickstart

List and copy packaged example configs:

```bash
llm-accel examples list
llm-accel examples write --output-dir configs
```

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

Run a config-defined sweep:

```bash
llm-accel bench sweep --config configs/benchmark_small.yaml
llm-accel bench sweep --config configs/benchmark_prompts.yaml
llm-accel bench sweep --config configs/benchmark_prefix_cache.yaml
```

Run the deterministic five-profile mock matrix:

```bash
llm-accel bench matrix \
  --config configs/optimization_matrix_mock.yaml \
  --output-dir results/runs/mock-optimization-matrix
```

The matrix covers baseline, prefix cache, chunked prefill, quantized, and speculative treatment profiles in randomized order for three repetitions.
It checkpoints `matrix_state.json` after every cell and resumes only when the config digest and existing run artifacts remain valid.
Mock matrix output proves orchestration and evidence gates only.
It is never model, backend, or hardware performance evidence.
Real matrices require one explicit, distinct, already-running endpoint URL per profile.
The tool does not provision hardware, download models, or silently restart serving processes.

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

Run a throughput-focused benchmark:

```bash
llm-accel bench throughput \
  --base-url mock://local \
  --model mock-model \
  --concurrency 4 \
  --input-tokens 128 \
  --output-tokens 64 \
  --request-count 8 \
  --output-dir results/runs/readme-throughput
```

Throughput runs preserve the standard raw request artifacts and add `throughput_summary.json` plus `throughput_summary.md`.

Estimate KV cache memory:

```bash
llm-accel kv-cache estimate \
  --preset llama-3-8b \
  --seq-len 8192 \
  --batch-size 16 \
  --dtype fp16 \
  --json
```

List built-in model-shape presets:

```bash
llm-accel kv-cache presets
```

Run the environment check:

```bash
llm-accel doctor
```

Run a lightweight quality sanity evaluation:

```bash
llm-accel eval sanity \
  --base-url mock://local \
  --model mock-model \
  --prompts configs/spec_prompts.jsonl \
  --output-dir results/runs/eval-smoke
```

Run a validator-based task evaluation:

```bash
llm-accel eval task \
  --base-url mock://local \
  --model mock-model \
  --tasks configs/task_eval_small.jsonl \
  --output-dir results/runs/task-eval-smoke
```

Validate and compare generated runs:

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
The claim audit intentionally rejects this mock smoke run because it is not hardware evidence.
The ranking audit also rejects the mock matrix, closed-loop coordinated-omission risk, client saturation, missing repetitions, missing quality deltas, and any corrupted source evidence.

Inspect backend capability metadata:

```bash
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

Validate vLLM benchmark readiness:

```bash
llm-accel vllm validate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --base-url http://localhost:8000/v1 \
  --output-dir results/runs/vllm-validation
```

Generate a hardware benchmark runbook:

```bash
llm-accel vllm plan \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --hardware-label GPU_CLASS \
  --dtype float16 \
  --base-url http://localhost:8000/v1 \
  --output-dir results/runs/vllm-plan
```

Run a mock quantization comparison:

```bash
llm-accel quantization compare \
  --base-url mock://local \
  --model mock-model \
  --modes none,int8,int4 \
  --output-dir results/runs/quantization-smoke
```

Requested quantization modes are labeled as `supported`, `unsupported`, or `unknown`; unsupported modes are reported but not benchmarked.

## Development

```bash
python3 scripts/release_check.py --metadata-only
python3 -m pip install -e . --dry-run
llm-accel --help
python3 -m ruff check .
python3 -m pytest
python3 scripts/smoke.py
```

The default test suite does not require a GPU.

See [docs/release.md](docs/release.md) for release and benchmark-claim checks.
See [docs/proposal_implementation_audit.md](docs/proposal_implementation_audit.md) for the current proposal-to-implementation coverage map.
See [docs/research_optimization_plan.md](docs/research_optimization_plan.md) for research-backed optimization directions.
See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) for contribution, conduct, and security guidance.

## Project Direction

See [proposal.md](proposal.md) for the full product proposal, architecture, benchmark methodology, roadmap, and open source model.
