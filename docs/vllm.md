# vLLM Workflow

The project targets vLLM as the first concrete serving backend.

Generate a server command:

```bash
llm-accel vllm command \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --enable-prefix-caching \
  --enable-chunked-prefill
```

Then run a benchmark against the OpenAI-compatible endpoint:

```bash
llm-accel bench latency \
  --base-url http://localhost:8000/v1 \
  --backend vllm \
  --model meta-llama/Llama-3.2-1B-Instruct
```

The helper prints commands only. It does not automatically start a long-running server process.

The endpoint client supports OpenAI-compatible streaming server-sent events and non-streaming JSON responses. Streaming mode records TTFT from the first content-bearing event.

## Optimization Flags

`vllm command`, `vllm validate`, and `vllm plan` accept the same startup-optimization flags:

- `--enable-prefix-caching`
- `--enable-chunked-prefill`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--speculative-model`
- `--num-speculative-tokens`

These flags are recorded in validation and plan artifacts through the generated startup command. They do not create a benchmark claim by themselves; compare runs only after the generated benchmark directories pass validation and the startup command is included with the results.

## Readiness Validation

Use `vllm validate` before claiming a hardware-backed benchmark:

```bash
llm-accel vllm validate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --base-url http://localhost:8000/v1 \
  --output-dir results/runs/vllm-validation
```

The validator checks:

- whether the `vllm` Python package is importable
- whether GPU telemetry is available through `nvidia-smi`
- whether the OpenAI-compatible endpoint is healthy
- optionally, whether a smoke completion succeeds with `--smoke`

Outputs:

- `manifest.json`
- `vllm_validation.json`
- `vllm_validation.md`

If the machine lacks vLLM or GPU telemetry, the validator records explicit blockers instead of fabricating benchmark readiness.

## Hardware Benchmark Plan

Generate a runbook for a GPU/vLLM machine:

```bash
llm-accel vllm plan \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --hardware-label GPU_CLASS \
  --dtype float16 \
  --base-url http://localhost:8000/v1 \
  --config configs/benchmark_vllm_small.yaml \
  --output-dir results/runs/vllm-plan
```

Outputs:

- `manifest.json`
- `vllm_benchmark_plan.json`
- `vllm_benchmark_plan.md`

The plan records the exact command sequence for validation, server startup, latency benchmark, throughput benchmark, sweep, task evaluation, and report validation. It also lists claim rules so benchmark results are not overstated.
