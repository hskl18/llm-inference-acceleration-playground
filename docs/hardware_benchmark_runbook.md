# Hardware Benchmark Runbook

This runbook collects real vLLM evidence on an NVIDIA GPU host.
The repository does not currently include a hardware result, so mock output must not be used as a performance claim.

## Target Experiment

The first publishable experiment compares two server profiles on the same host, model revision, dtype, prompt set, and client settings:

1. Baseline vLLM serving.
2. Automatic prefix caching with a shared-prefix workload.

Each profile uses at least three repetitions, eight warmup requests, and 128 measured requests per repetition.
The collector writes raw JSONL, CSV, manifests, summaries, plots, per-run claim audits, and a repeated-run aggregate.

## Record Before Starting

- GPU model and memory capacity.
- NVIDIA driver and CUDA version.
- Python, PyTorch, and vLLM versions.
- Immutable full hexadecimal model revision, not a branch or tag such as `main`.
- Exact vLLM startup command.
- Repository commit.
- Host label that identifies the hardware class without exposing private infrastructure names.

The benchmark runner captures the available software and accelerator fields in every `summary.json`.
The model revision and optimization profile must be supplied explicitly.

## Install on the GPU Host

Use an environment compatible with the selected vLLM and CUDA versions, then install this repository:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install vllm
```

Do not copy API keys into a config or result directory.
The local vLLM endpoint does not require an API key unless the operator configures one.

## Profile A: Baseline

In terminal A, generate the server command record and start the blocking server process.
Keep terminal A open while the collector runs in terminal B.

```bash
llm-accel vllm command \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --dtype float16 \
  > baseline-server-command.txt
bash baseline-server-command.txt
```

Validate readiness:

```bash
llm-accel vllm validate \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --base-url http://localhost:8000/v1 \
  --dtype float16 \
  --smoke \
  --output-dir results/hardware-v1/baseline-validation
```

Collect repeated evidence:

```bash
python scripts/run_hardware_profile.py \
  --profile baseline \
  --model MODEL_ID \
  --model-revision MODEL_REVISION \
  --hardware-label GPU_CLASS \
  --dtype float16 \
  --prompts configs/prefix_cache_prompts.jsonl \
  --server-command-file baseline-server-command.txt \
  --output-root results/hardware-v1
```

## Profile B: Prefix Cache

Stop the baseline server in terminal A, confirm the process has exited, and restart with prefix caching enabled:

```bash
llm-accel vllm command \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --dtype float16 \
  --enable-prefix-caching \
  > prefix-cache-server-command.txt
bash prefix-cache-server-command.txt
```

Validate the restarted profile into a separate directory:

```bash
llm-accel vllm validate \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --base-url http://localhost:8000/v1 \
  --dtype float16 \
  --enable-prefix-caching \
  --smoke \
  --output-dir results/hardware-v1/prefix-cache-validation
```

Then collect the second profile in terminal B:

```bash
python scripts/run_hardware_profile.py \
  --profile prefix-cache \
  --model MODEL_ID \
  --model-revision MODEL_REVISION \
  --hardware-label GPU_CLASS \
  --dtype float16 \
  --prompts configs/prefix_cache_prompts.jsonl \
  --server-command-file prefix-cache-server-command.txt \
  --output-root results/hardware-v1
```

## Cross-Profile Comparison

Compare all six repetition summaries only after both profile collectors pass:

```bash
llm-accel report compare \
  --summary results/hardware-v1/baseline/repeat-01/summary.json \
  --summary results/hardware-v1/baseline/repeat-02/summary.json \
  --summary results/hardware-v1/baseline/repeat-03/summary.json \
  --summary results/hardware-v1/prefix-cache/repeat-01/summary.json \
  --summary results/hardware-v1/prefix-cache/repeat-02/summary.json \
  --summary results/hardware-v1/prefix-cache/repeat-03/summary.json \
  --output-dir results/hardware-v1/comparison
```

The comparison report groups repetitions by optimization profile and reports mean, standard deviation, minimum, and maximum throughput and p95 latency.
Inspect raw request failures and the full distributions before publishing a relative result.

## Claim Audit

Audit each run independently:

```bash
llm-accel report claim-audit --run-dir results/hardware-v1/baseline/repeat-01
```

The audit rejects mock or relabeled runs, missing artifacts, a missing or changed server command record, mutable model revisions, missing GPU or software versions, fewer than 100 completed requests, fewer than five warmups, excessive errors, invalid p50/p95/p99 latency fields, invalid throughput, unavailable GPU memory telemetry, and raw-request inconsistencies.
Passing this audit is necessary but not sufficient for a portfolio claim.
The audit cannot independently attest which process served the endpoint, so preserve the platform launch record or operator evidence with published results.

## Quality and Interpretation

Run the same task fixture against both server profiles and report the quality score beside performance results.
Prefix caching should preserve output behavior, but the repository does not infer that from speed measurements.
Keep request failures, timeout rate, error rate, output token count, GPU memory, and every warning visible.
Do not compare profiles when the model revision, hardware, dtype, quantization, workload fingerprint, token lengths, or request settings differ.

Collect the baseline task artifact while the baseline server is running:

```bash
llm-accel eval task \
  --base-url http://localhost:8000/v1 \
  --backend vllm \
  --model MODEL_ID \
  --tasks configs/task_eval_small.jsonl \
  --stream \
  --output-dir results/hardware-v1/baseline-quality
```

Run the same command after the prefix-cache restart, changing only the output directory to `results/hardware-v1/prefix-cache-quality`.

## Publication Checklist

- All per-run claim audits pass.
- Each profile has at least three valid repetitions.
- The comparison has no compatibility or low-sample warnings.
- Raw JSONL and manifests are included.
- The exact model revision and vLLM command are recorded.
- p50, p95, and p99 TTFT, TPOT, and end-to-end latency are reported.
- Output tokens/sec, requests/sec, error rate, timeout rate, and GPU memory are reported.
- Quality evidence is reported separately and any delta is disclosed.
- The result states the GPU host and software environment.
- Limitations and non-comparable runs remain visible.
