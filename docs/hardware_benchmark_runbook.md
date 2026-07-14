# Hardware Benchmark Runbook

This runbook describes the evidence required for a real vLLM optimization matrix on an NVIDIA GPU host.
The repository does not include a real GPU result as of version 0.2.0.
Mock results validate the workflow and evidence gates only.

## Experiment Boundary

The v0.2.0 matrix covers five treatment profiles:

1. Baseline serving.
2. Prefix caching.
3. Chunked prefill.
4. Quantized serving.
5. Speculative decoding.

Every profile requires at least three measured repetitions.
Profile order is randomized within each repetition.
Every measured cell runs its own warmups before raw request collection begins.

The matrix runner does not provision hardware, install vLLM, download models, or restart servers.
Real matrices require one explicit and distinct already-running endpoint URL per profile.
This keeps server lifecycle and billable infrastructure outside the benchmark process.

## Required Host Evidence

Record these fields before collection:

- GPU model and memory capacity.
- NVIDIA driver and CUDA versions.
- Python, PyTorch, and vLLM versions.
- Full immutable target-model revision.
- Full immutable tokenizer revision.
- Full immutable speculative draft-model revision when applicable.
- Exact startup command for every endpoint.
- Repository commit.
- Hardware label that identifies the measured class without exposing a private hostname.

Every endpoint must use the same target model, target revision, tokenizer revision, hardware class, and software environment.
Only the declared optimization treatment may differ.

## Install on an Authorized GPU Host

Use an environment compatible with the selected vLLM and CUDA versions:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install vllm
```

Do not copy API keys into a config or result directory.
Do not start billable hardware or use a paid endpoint without separate authorization.

## Launch and Validate Each Profile

Generate each command with `llm-accel vllm command`, preserve the exact output as a command file, and launch each profile on its own port or host.
The examples below omit model-specific quantization and speculative arguments that must be selected for the authorized model and hardware.

Baseline example:

```bash
llm-accel vllm command \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --dtype float16 \
  --port 8000 \
  > baseline-server-command.txt
```

Prefix-cache example:

```bash
llm-accel vllm command \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --dtype float16 \
  --port 8001 \
  --enable-prefix-caching \
  > prefix-cache-server-command.txt
```

Chunked-prefill example:

```bash
llm-accel vllm command \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --dtype float16 \
  --port 8002 \
  --enable-chunked-prefill \
  > chunked-prefill-server-command.txt
```

Validate each live endpoint before the matrix run:

```bash
llm-accel vllm validate \
  --model MODEL_ID \
  --revision MODEL_REVISION \
  --dtype float16 \
  --base-url http://localhost:8000/v1 \
  --smoke \
  --output-dir results/hardware-v0.2/validation/baseline
```

Repeat validation with the exact flags and URL for every treatment.
Do not proceed when any validation artifact reports blockers.

## Matrix Configuration

Start from `configs/optimization_matrix_mock.yaml`, then replace mock identity, endpoints, command evidence, quality tasks, workload shape, and hardware label.
Keep at least three repetitions, at least five warmups, and at least 100 measured requests per cell.

Use open-loop scheduling for ranking evidence:

```yaml
workload:
  mode: fixed_prompts
  prompts_path: prefix_cache_prompts.jsonl
  request_schedule: open-loop
  request_rate_rps: 20
  input_tokens: [512]
  output_tokens: [128]
  concurrency: [8]
  seed: 42
```

Set `run.client_processes` high enough to avoid client saturation without exceeding total concurrency.
Choose the offered request rate from a no-claim calibration run.
If queue-delay p95 exceeds the stricter of the declared threshold and the audit's request-cadence ceiling, lower the offered rate or increase authorized client capacity and rerun every affected cell.

Every real profile mapping must provide its own `base_url` and `server_command_file`.
The profile fields must match the exact command flags, including quantization, prefix cache, chunked prefill, speculative model, speculative token count, batching limits, model length, and GPU-memory utilization.

## Run and Resume

Run the matrix:

```bash
llm-accel bench matrix \
  --config configs/hardware_matrix.yaml \
  --output-dir results/hardware-v0.2/matrix
```

If the process stops, resume with the unchanged config:

```bash
llm-accel bench matrix \
  --config configs/hardware_matrix.yaml \
  --output-dir results/hardware-v0.2/matrix \
  --resume
```

Resume rejects a changed config digest.
It skips only cells whose profile artifact, manifest, summary, and listed artifacts still validate.
Stale running cells return to pending state and are retried.

## Inspect Evidence

Matrix cells include a structured optimization profile with exact command and tokenizer identity.
Standalone commands generated by `llm-accel vllm plan` also bind the tokenizer name and immutable tokenizer revision into benchmark metadata.

Validate the matrix root and representative cells:

```bash
llm-accel report validate --run-dir results/hardware-v0.2/matrix
llm-accel report validate --run-dir results/hardware-v0.2/matrix/baseline/repeat-01/c8-in512-out128
```

Audit a single hardware run:

```bash
llm-accel report claim-audit \
  --run-dir results/hardware-v0.2/matrix/baseline/repeat-01/c8-in512-out128
```

Audit the complete performance ranking:

```bash
llm-accel report ranking-audit \
  --matrix-dir results/hardware-v0.2/matrix
```

## Publication Gate

Do not publish a profile ranking unless all of these conditions hold:

- Every source run passes the single-run hardware audit.
- Every profile has at least three valid repetitions.
- The comparison contains exactly one compatible invariant stratum.
- Raw request traces reproduce summary latency, throughput, queue delay, and request counts.
- Exact command files match structured optimization profiles and server flags.
- Target model, tokenizer, prompt, schedule, client, quality gate, environment, request shape, and code invariants match.
- Quality results exist for every profile and disclose score deltas from baseline.
- Every quality gate passes.
- Open-loop scheduled-versus-dispatch evidence is complete.
- Client queue-delay p95 stays within the effective saturation threshold derived from the declared threshold and offered request cadence.
- Execution failures, endpoint request failures, timeouts, and interrupted repetitions remain disclosed.
- The platform or operator launch record supports the recorded server command for every endpoint.

The ranking audit checks repository artifacts but cannot independently attest which live process served an endpoint.
Preserve platform launch logs or an operator record with any published result.
