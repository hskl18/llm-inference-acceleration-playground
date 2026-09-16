# Quantization

A quantization mode is a property of the weights a server loaded, not of a request.
One endpoint therefore cannot represent two modes, and `quantization compare` requires one already-running endpoint per mode:

```bash
llm-accel quantization compare \
  --model MODEL_ID \
  --backend vllm \
  --mode none=http://localhost:8000/v1 \
  --mode awq=http://localhost:8001/v1 \
  --mode fp8=http://localhost:8002/v1 \
  --output-dir results/runs/quantization-comparison
```

The first `--mode` is the baseline.
Two modes that resolve to the same endpoint are rejected rather than relabelled, so an unchanged server can no longer be reported as three different modes.

Each endpoint is also asked for `GET /models`; a mode whose endpoint does not list the requested model raises a warning that the served model identity could not be confirmed.

## Mode Names

Use the backend's own method names.
For vLLM these come from `QuantizationMethods` in `vllm/model_executor/layers/quantization/__init__.py` (v0.29.0): `awq`, `auto_awq`, `awq_marlin`, `gptq`, `auto_gptq`, `gptq_marlin`, `fp8`, `compressed-tensors`, `modelopt`, `modelopt_fp4`, `quark`, `torchao`, `experts_int8`, `moe_wna16`, `mxfp4`, and others.
Plain `int8` and `int4` are not vLLM modes and are rejected when a command is generated.

TensorRT-LLM selects quantization when the engine is built rather than through a serving flag, so its capability entry reports `unknown`.

Each requested mode is labeled with a support status:

- `supported`: listed in the backend capability matrix and benchmarked.
- `unsupported`: not listed for the backend; reported but not benchmarked.
- `unknown`: backend capabilities are not known locally; benchmark output is endpoint-defined.

Unsupported modes do not produce throughput or latency claims.

## Quality Evidence

Reports should separate:

- memory savings
- TTFT changes
- TPOT and throughput changes
- p95 latency changes
- output quality against the baseline mode

Quantization changes the weights, so an output-quality check that only asserts non-empty text proves nothing.
The comparison scores every mode against the baseline mode on the same fixed prompts:

- **Temperature-0 exact-match rate**: the fraction of sanity prompts where the mode reproduces the baseline mode's output exactly. A rate below 1.0 raises a warning.
- **Perplexity delta**: perplexity of one fixed text, computed from echoed prompt logprobs (`/v1/completions` with `echo` and `logprobs`), minus the baseline mode's perplexity.

Not every backend returns prompt logprobs. When a mode's endpoint does not, the run records `prompt_logprobs_supported: false` with the error and reports no perplexity delta rather than guessing one.

Both checks are lightweight screens. They do not replace a task-specific evaluation such as `llm-accel eval task`.

The mock backend validates the workflow and report shape only. Real quantization claims require a backend that actually runs each mode on its own server.
