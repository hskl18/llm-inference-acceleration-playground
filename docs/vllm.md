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

That prints a `vllm serve <model>` command line.
`vllm serve` is the documented entrypoint (https://docs.vllm.ai/en/latest/cli/serve.html).
The older `python -m vllm.entrypoints.openai.api_server` module is deprecated in vLLM and emits a `DeprecationWarning` that points at `vllm.entrypoints.launchers` (vLLM v0.29.0, `vllm/entrypoints/openai/api_server.py`), so this project no longer generates or accepts it.

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

- `--enable-prefix-caching` / `--no-enable-prefix-caching`
- `--enable-chunked-prefill` / `--no-enable-chunked-prefill`
- `--max-num-batched-tokens`
- `--max-num-seqs`
- `--speculative-method`
- `--speculative-model`
- `--num-speculative-tokens`

### Prefix caching and chunked prefill are always stated explicitly

vLLM turns both features on by default: `CacheConfig.enable_prefix_caching: bool = True` (`vllm/config/cache.py`) and `SchedulerConfig.enable_chunked_prefill: bool = True` (`vllm/config/scheduler.py`), both in vLLM v0.29.0.
An omitted flag therefore does not mean "off", and a baseline that simply leaves the flag out is not a baseline.

Generated commands always contain one of `--enable-prefix-caching` or `--no-enable-prefix-caching`, and one of `--enable-chunked-prefill` or `--no-enable-chunked-prefill`.
vLLM accepts both spellings because it binds boolean configuration fields with `argparse.BooleanOptionalAction` (`vllm/engine/arg_utils.py`).
`report claim-audit` rejects a recorded server command that leaves either feature implicit.

### Speculative decoding uses one JSON object

Current vLLM has no `--speculative-model` or `--num-speculative-tokens` flag.
Speculative decoding is configured through a single `--speculative-config` JSON object (https://docs.vllm.ai/en/latest/features/speculative_decoding/):

```bash
llm-accel vllm command \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dtype float16 \
  --speculative-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --num-speculative-tokens 5
```

emits

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct ... \
  --speculative-config '{"method":"draft_model","model":"HuggingFaceTB/SmolLM2-135M-Instruct","num_speculative_tokens":5}'
```

`--speculative-method` defaults to `draft_model` when a draft model is given.
Methods that do not use a separate checkpoint, such as `ngram`, need `--speculative-method` and `--num-speculative-tokens` but no `--speculative-model`.
The JSON is emitted with sorted-free, fixed key order and no spaces so the command text, and therefore its SHA-256, is reproducible.

### Quantization method names

`--quantization` takes a vLLM method name, not a bit width.
Valid values come from `QuantizationMethods` in `vllm/model_executor/layers/quantization/__init__.py` (vLLM v0.29.0) and include `awq`, `gptq`, `gptq_marlin`, `awq_marlin`, `fp8`, `compressed-tensors`, `modelopt`, `quark`, `torchao`, and `experts_int8`.
Plain `int8` and `int4` are not vLLM method names.

vLLM itself declares `--quantization` with a `metavar` rather than argparse `choices`, so a misspelled value is only rejected when the engine loads the model.
This project validates the name before it writes a command, and `bench matrix` validates `profiles.<name>.quantization` for any profile whose backend is `vllm`.

These flags do not create a benchmark claim by themselves; compare runs only after the generated benchmark directories pass validation and the startup command is included with the results.

## Readiness Validation

Use `vllm validate` before claiming a hardware-backed benchmark:

```bash
llm-accel vllm validate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --revision MODEL_REVISION \
  --base-url http://localhost:8000/v1 \
  --output-dir results/runs/vllm-validation
```

A benchmark client usually runs on a different machine than the server, so the validator asks the server about itself:

- `GET /version` returns `{"version": ...}` and is the recorded backend version.
- `GET /metrics` returns the Prometheus exposition, including `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`, and the speculative counters `vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`, and `vllm:spec_decode_num_accepted_tokens_total`.
- `GET /v1/models` decides endpoint health.

Blockers are raised when the endpoint is not healthy, when `/version` does not identify the serving process, or when a requested `--smoke` completion fails.

A locally importable `vllm` package and a local `nvidia-smi` describe the client host, not the endpoint, so they are reported as informational warnings.
Pass `--same-host` when the client and the server really are the same machine; then both become blockers again.

Outputs:

- `manifest.json`
- `vllm_validation.json`
- `vllm_validation.md`

Non-local endpoint URLs are redacted in every written artifact.

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

Every written step command uses the same endpoint redaction as the rest of the project.
A non-local `--base-url` appears as `redacted` in the plan and in each step command; substitute the real endpoint when running the steps.
