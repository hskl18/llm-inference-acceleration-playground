# LLM Inference Acceleration Playground: Project Proposal

## 1. Overview

LLM Inference Acceleration Playground is an open source project for measuring, comparing, and explaining inference-serving performance tradeoffs for large language models. It focuses on practical benchmarking rather than building a new inference engine.

The project provides a reproducible benchmark harness, analysis tools, and reference reports for OpenAI-compatible LLM serving endpoints. The initial target backend is vLLM, with the architecture kept flexible enough to add other serving systems later.

The core product is a developer-facing toolkit that helps answer questions such as:

- How does concurrency affect time to first token, total latency, and throughput?
- Where does batching improve throughput, and where does it harm tail latency?
- How much GPU memory is consumed by KV cache for a given model shape and workload?
- Does quantization improve serving speed, or does it mainly reduce memory usage?
- Under what acceptance-rate conditions does speculative decoding actually help?

## 2. Problem Statement

Inference acceleration discussions often mix together kernel speed, model architecture, serving policy, memory pressure, and benchmark methodology. That makes it difficult to tell whether an optimization improves end-to-end serving behavior.

This project treats inference acceleration as a systems measurement problem. Each experiment should record the workload, serving backend, model configuration, hardware environment, raw metrics, and generated analysis so results are inspectable and reproducible.

## 3. Project Goals

- Provide a clean benchmark harness for OpenAI-compatible completion and chat-completion endpoints.
- Measure streaming and non-streaming inference metrics, including TTFT, TPOT, total latency, throughput, request rate, and tail latency.
- Run controlled sweeps over concurrency, prompt length, output length, and serving configuration.
- Estimate KV cache memory usage from model architecture and workload parameters.
- Compare baseline and quantized serving modes when the local backend supports them.
- Include a toy speculative decoding module for studying acceptance rate, draft cost, verification cost, and speedup.
- Generate reports that combine raw measurements, summary tables, plots, and concise engineering interpretation.
- Keep the codebase approachable for contributors who want to add backends, metrics, experiments, or reports.

## 4. Non-Goals

- This project will not implement a production LLM serving runtime.
- This project will not claim universal benchmark results from one machine or one model.
- This project will not hide unsupported hardware or backend features behind fake benchmark numbers.
- This project will not use average latency alone as a performance conclusion.
- This project will not treat paper notes as a substitute for runnable experiments.

## 5. Intended Users

- Engineers evaluating LLM serving performance on local or remote hardware.
- Students and researchers studying inference acceleration tradeoffs.
- Open source contributors who want a small, concrete codebase for serving benchmarks.
- Teams comparing backend configuration changes before adopting them in larger systems.

## 6. Product Scope

The project should ship as a command-line toolkit with a documented Python package structure. Users should be able to install the project, point it at an OpenAI-compatible endpoint, run a benchmark, and receive machine-readable results plus a human-readable report.

### Operating Modes

- **Local smoke mode**: runs against a mock or lightweight endpoint without requiring a GPU. This mode is used for tests, examples, and contributor onboarding.
- **Endpoint benchmark mode**: runs against any OpenAI-compatible endpoint supplied by the user.
- **vLLM benchmark mode**: includes vLLM-specific metadata, startup helpers, and backend capability checks.
- **Analysis-only mode**: reads existing result files and regenerates summaries, plots, and comparison reports without re-running inference.

### MVP User Flow

1. Start a serving backend, such as a vLLM OpenAI-compatible server.
2. Run a benchmark command with model, endpoint, workload, and concurrency options.
3. Inspect raw JSONL or CSV metrics.
4. Generate a Markdown report and optional plots.
5. Repeat with a different concurrency level, prompt length, model, dtype, or quantization mode.

### Example Commands

```bash
llm-accel bench latency \
  --base-url http://localhost:8000/v1 \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --concurrency 8 \
  --input-tokens 512 \
  --output-tokens 128 \
  --output-dir results/runs/local-baseline
```

```bash
llm-accel bench sweep \
  --config configs/benchmark_small.yaml \
  --output-dir results/runs/small-sweep
```

```bash
llm-accel kv-cache estimate \
  --layers 32 \
  --seq-len 8192 \
  --batch-size 16 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype fp16
```

```bash
llm-accel speculative run \
  --draft-model small-draft-model \
  --target-model larger-target-model \
  --lookahead 4 \
  --prompts configs/spec_prompts.jsonl \
  --output-dir results/runs/speculative-toy
```

### Public CLI Surface

The first stable CLI should expose these commands:

| Command | Purpose | Required output |
| --- | --- | --- |
| `llm-accel bench latency` | Run a single latency-focused workload | raw requests, summary JSON, summary Markdown |
| `llm-accel bench throughput` | Run a throughput-focused workload | raw requests, throughput summary |
| `llm-accel bench sweep` | Run a config-defined experiment matrix | one run directory per experiment plus aggregate report |
| `llm-accel report generate` | Regenerate reports from existing raw results | Markdown report and optional plots |
| `llm-accel kv-cache estimate` | Estimate KV cache memory for a model/workload shape | console output and optional JSON |
| `llm-accel speculative run` | Run the toy speculative decoding study | acceptance-rate report |
| `llm-accel doctor` | Check environment, optional GPU access, dependencies, and endpoint health | diagnostic report |

## 7. Technical Architecture

The codebase should separate serving clients, workload generation, benchmark execution, metric aggregation, plotting, and report generation. This keeps backend integrations and experiment logic testable in isolation.

### Core Modules

- `serving`: OpenAI-compatible clients, backend metadata collection, server startup helpers, and optional backend adapters.
- `workloads`: Prompt generation, fixed prompt sets, synthetic token-length controls, and dataset-backed workloads.
- `benchmarks`: Latency benchmark, throughput benchmark, concurrency sweep, sequence-length sweep, and benchmark orchestration.
- `metrics`: Timing capture, percentile calculation, token accounting, GPU memory sampling, and run metadata schemas.
- `kv_cache`: KV cache estimator, model-shape presets, and explanatory documentation.
- `quantization`: Quantized-backend experiment wrappers and result comparison helpers.
- `speculative_decoding`: Toy speculative decoding implementation and acceptance-rate analysis.
- `reports`: Markdown report generation, plot generation, and result summarization.
- `configs`: Reproducible benchmark and report configurations.

### Proposed Repository Structure

```text
llm-inference-acceleration-playground/
  README.md
  LICENSE
  pyproject.toml
  proposal.md
  configs/
    benchmark_small.yaml
    benchmark_local.yaml
    benchmark_prompts.yaml
    spec_prompts.jsonl
  src/
    llm_accel/
      cli.py
      serving/
        openai_client.py
        vllm.py
        metadata.py
      workloads/
        synthetic.py
        prompts.py
      benchmarks/
        latency.py
        throughput.py
        sweep.py
      metrics/
        timing.py
        aggregation.py
        memory.py
        schemas.py
      kv_cache/
        estimator.py
        presets.py
      config/
        loader.py
        schemas.py
      quantization/
        benchmark.py
        comparison.py
      speculative_decoding/
        vanilla.py
        analysis.py
      reports/
        markdown.py
        plots.py
  docs/
    benchmark_methodology.md
    kv_cache.md
    quantization.md
    speculative_decoding.md
    methods/
      flashattention.md
      medusa.md
      eagle.md
      specinfer.md
  tests/
    unit/
    integration/
  results/
    .gitkeep
```

### Packaging and Dependency Strategy

The project should use a standard Python package layout with `src/llm_accel`. The first implementation should target Python 3.10 or newer.

Recommended dependency groups:

- `core`: CLI, HTTP client, config parsing, schemas, and basic report generation.
- `plot`: plotting libraries for PNG or SVG output.
- `dev`: tests, linting, type checking, and mock server tools.
- `vllm`: optional vLLM-specific helpers where installation is practical.
- `gpu`: optional GPU telemetry dependencies.

The default installation should stay lightweight:

```bash
pip install llm-inference-acceleration-playground
```

Optional extras can be added later:

```bash
pip install "llm-inference-acceleration-playground[plot,dev]"
```

## 8. Configuration Model

Benchmark runs should be configurable from both CLI flags and YAML files. CLI flags are useful for quick experiments; YAML files are required for reproducible sweeps.

### Example Sweep Configuration

```yaml
run:
  name: small-local-sweep
  output_dir: results/runs/small-local-sweep
  warmup_requests: 4
  measured_requests: 32
  timeout_seconds: 120

endpoint:
  base_url: http://localhost:8000/v1
  api_key_env: OPENAI_API_KEY
  backend: vllm

model:
  name: meta-llama/Llama-3.2-1B-Instruct
  dtype: fp16
  quantization: none

workload:
  mode: synthetic
  input_tokens: [128, 512]
  output_tokens: [64, 256]
  concurrency: [1, 4, 8]
  seed: 42

report:
  formats: [markdown, json]
  plots: true
```

Fixed-prompt sweeps can replace `workload.input_tokens` with a prompt file:

```yaml
workload:
  mode: prompts
  prompts_path: configs/spec_prompts.jsonl
  output_tokens: [64]
  concurrency: [1, 4]
```

### Configuration Requirements

- Config parsing must fail with clear validation errors.
- Defaults must be documented and visible in generated metadata.
- Secrets must be referenced through environment variable names, not written into result files.
- The final resolved config should be saved with each benchmark run.

## 9. Benchmarking Methodology

Benchmarks should be designed so that results can be interpreted rather than merely displayed.

### Workload Dimensions

The initial sweep matrix should cover:

```text
concurrency:   1, 2, 4, 8, 16, 32
input tokens:  128, 512, 2048
output tokens: 64, 256, 1024
```

The default configuration should be small enough to run on accessible hardware. Larger sweeps should be opt-in.

### Required Metrics

- TTFT: time from request start to first generated token.
- TPOT: time per output token after the first token.
- Total latency: time from request start to final token.
- Output tokens/sec: generated tokens divided by wall-clock time.
- Requests/sec: completed requests divided by wall-clock time.
- p50, p95, and p99 latency.
- Error rate and timeout count.
- Peak or sampled GPU memory usage when available.
- Run metadata, including model, backend, dtype, quantization mode, hardware label, and timestamp.

### Raw Result Format

Per-request records should be written as JSONL so large runs can be streamed and inspected incrementally.

Example record:

```json
{
  "request_id": "req-000001",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "backend": "vllm",
  "input_tokens": 512,
  "output_tokens": 128,
  "concurrency": 8,
  "ttft_ms": 142.8,
  "tpot_ms": 18.4,
  "total_latency_ms": 2479.5,
  "completed": true,
  "error": null
}
```

### Summary Output

Each run should generate:

- `raw_requests.jsonl`
- `raw_requests.csv`
- `resolved_config.json`
- `run_metadata.json`
- `summary.json`
- `summary.md`
- `plots/latency.svg`
- `throughput_summary.json` and `throughput_summary.md` for throughput-focused runs

## 10. Data Model and Result Artifacts

The project should treat benchmark output as a stable product surface. Reports can evolve, but raw result records and summary schemas should be versioned.

### Run Directory Layout

```text
results/runs/2026-05-26T12-30-00-small-local-sweep/
  resolved_config.json
  run_metadata.json
  raw_requests.jsonl
  raw_requests.csv
  summary.json
  summary.md
  plots/
    latency.svg
```

### Summary Schema

Each `summary.json` should include:

- schema version
- run metadata
- request counts
- error and timeout counts
- token counts
- latency percentiles
- throughput metrics
- memory metrics when available
- warnings about unsupported or missing measurements

### Report Requirements

- Reports must clearly label measured data versus estimates.
- Reports must mention warmup count and measured request count.
- Reports must include failed-request counts.
- Reports must avoid ranking configurations when the run was too small or too noisy to justify a conclusion.

## 11. KV Cache Estimator

The KV cache module should provide a small, reliable estimator that explains memory growth under long context and high concurrency.

Formula:

```text
KV cache bytes =
  layers * sequence_length * batch_size * 2 * kv_heads * head_dim * dtype_bytes
```

The estimator must accept explicit `kv_heads` and `head_dim` values so it can represent multi-head attention, multi-query attention, and grouped-query attention.

The output should include:

- raw bytes
- MiB
- GiB
- model/workload parameters
- short explanation of the dominant scaling factors

## 12. Quantization Plan

Quantization support should be benchmarked only when the backend and hardware can actually run it. The project should distinguish measured results from documentation notes.

Initial comparison targets:

- FP16 or BF16 baseline
- 8-bit mode if supported
- 4-bit mode if supported
- AWQ or GPTQ backend notes when measured runs are unavailable

The quantization report should answer:

- How much memory was saved?
- Did TTFT change?
- Did TPOT or throughput change?
- Did p95 latency improve or regress?
- Did the output pass a small fixed-prompt sanity check?

## 13. Speculative Decoding Module

The speculative decoding module is a teaching and measurement component, not a production implementation. Its purpose is to make acceptance rate and verification cost concrete.

The toy implementation should:

1. Generate `k` draft tokens.
2. Verify draft tokens with the target model.
3. Accept matching or high-probability tokens according to the selected policy.
4. Fall back to target generation after rejection.
5. Record draft calls, target calls, accepted tokens, rejected tokens, acceptance rate, and wall-clock time.

The report should explain why speculative decoding may fail to speed up when:

- the draft model is too weak
- the draft model is too expensive
- target verification is not efficient enough
- acceptance rate is low
- batching or serving overhead dominates generation time

## 14. Privacy, Safety, and Benchmark Integrity

The project should be safe to run on local and remote endpoints without leaking secrets or producing misleading benchmark claims.

Requirements:

- API keys and endpoint credentials must never be written to result artifacts.
- Prompt datasets should be treated as user-provided data and should not be uploaded anywhere by default.
- Result files should include endpoint host metadata only when the user allows it or when it is already a local URL.
- Failed requests, retries, and timeouts must be visible in summaries.
- Benchmark reports must not claim superiority from non-comparable runs.
- Sample results committed to the repository must be clearly labeled as examples, not authoritative performance claims.

## 15. Documentation Plan

The documentation should support both users and contributors.

Required docs:

- `README.md`: project purpose, quickstart, installation, basic benchmark command, and result example.
- `docs/benchmark_methodology.md`: definitions, workload design, noise control, and interpretation guidance.
- `docs/result_schemas.md`: raw request, summary, memory, and aggregate artifact schemas.
- `docs/backend_capabilities.md`: known backend capabilities and limitations.
- `docs/vllm.md`: vLLM command helper and benchmark workflow.
- `docs/quality_eval.md`: fixed-prompt output sanity evaluation and keyword-rubric task evaluation.
- `docs/kv_cache.md`: estimator formula, examples, and limitations.
- `docs/quantization.md`: supported modes, backend limitations, and reporting rules.
- `docs/speculative_decoding.md`: toy algorithm, acceptance rate, and limitations.
- `docs/methods/*.md`: concise notes for FlashAttention, Medusa, EAGLE, and SpecInfer.
- `CONTRIBUTING.md`: development setup, test commands, issue labels, and contribution expectations.
- `CHANGELOG.md` and `docs/release.md`: release notes, validation checklist, and benchmark-claim rules.

## 16. Testing and Validation

The project should include tests from the first implementation milestone.

### Unit Tests

- KV cache formula and dtype conversion.
- Percentile and throughput aggregation.
- Prompt/workload generation boundaries.
- Result schema validation.
- Report generation from fixed sample data.

### Integration Tests

- Mock OpenAI-compatible streaming endpoint.
- Benchmark run against the mock endpoint.
- Sweep config parsing and result directory generation.
- CLI smoke tests for all public commands.

### Validation Rules

- Raw metrics must be saved even if report generation fails.
- Failed requests must be counted and reported, not silently dropped.
- Benchmark metadata must include enough information to reproduce the run.
- Tests should not require a GPU by default.

### Acceptance Gates by Milestone

| Milestone | Required verification |
| --- | --- |
| Foundation | `llm-accel --help` works, package imports cleanly, unit tests run |
| Baseline benchmarking | mock endpoint integration test produces raw requests and summary report |
| Sweeps and reports | config-defined sweep creates separate run directories and aggregate report |
| KV cache | known formula fixtures pass and CLI emits JSON-compatible output |
| Quantization | unsupported modes are reported clearly; supported modes record metadata |
| Speculative decoding | fixed toy prompts produce deterministic acceptance-rate accounting |

## 17. Reproducibility Requirements

Every benchmark run should record:

- project version
- git commit when available
- timestamp
- operating system
- Python version
- backend name and version when available
- model name
- dtype
- quantization mode
- endpoint base URL
- concurrency
- input length
- workload mode and prompt-set fingerprint for fixed-prompt runs
- output length
- warmup count
- request count
- timeout settings
- hardware label and GPU name when available

Results should be stored in timestamped run directories under `results/runs/`. Large raw outputs should be ignored by git by default, while small sample outputs can be committed for documentation and tests.

## 18. Release and Versioning Strategy

The project should use semantic versioning once a CLI and result schema are published.

- `0.x` releases can change CLI flags and schemas while the project is still stabilizing.
- Result schemas should include explicit `schema_version` fields from the first implementation.
- Breaking schema changes should include migration notes or compatibility readers when practical.
- Release notes should mention new metrics, schema changes, backend support, and known limitations.

## 19. Open Source Model

The project should follow standard open source practices from the start.

### License

Use a permissive license unless there is a reason to restrict downstream use. The initial repo uses MIT for a lightweight open source default. Apache-2.0 can be reconsidered before a public release if explicit patent language becomes important.

### Contribution Style

Contributions should be organized around small, reviewable changes:

- new backend adapter
- new metric
- new benchmark config
- new report template
- new method note
- bug fix with regression test

### Issue Labels

Suggested labels:

- `backend`
- `benchmark`
- `documentation`
- `good first issue`
- `kv-cache`
- `metrics`
- `quantization`
- `speculative-decoding`
- `testing`

### Project Quality Bar

Before accepting substantial changes, the project should require:

- passing tests
- type checking or linting once configured
- documented CLI behavior
- sample output or fixture updates when result schemas change
- no fabricated benchmark results

## 20. Roadmap

### Milestone 0: Project Foundation

- Create package structure.
- Add CLI skeleton.
- Add config loading.
- Add result directory conventions.
- Add unit test setup.
- Add README and contribution guide.
- Add license and code of conduct decision.

### Milestone 1: Baseline Benchmarking

- Implement OpenAI-compatible client.
- Implement latency benchmark.
- Capture streaming TTFT and total latency.
- Generate raw JSONL and summary Markdown.
- Add mock-server integration tests.
- Add `llm-accel doctor` endpoint check.

### Milestone 2: Sweeps and Reports

- Add concurrency and sequence-length sweep runner.
- Add summary aggregation.
- Add plot generation.
- Add benchmark methodology docs.
- Add resolved-config persistence.

### Milestone 3: KV Cache and Memory Analysis

- Implement KV cache estimator.
- Add model-shape presets.
- Add memory-report section.
- Add examples for long-context workloads.
- Add JSON output mode for automation.

### Milestone 4: Quantization Comparisons

- Add quantization benchmark metadata.
- Support backend-specific quantization configuration.
- Generate baseline-versus-quantized comparison reports.
- Document unsupported modes clearly.
- Add fixed-prompt sanity check output.

### Milestone 5: Speculative Decoding Study

- Implement toy speculative decoding.
- Add acceptance-rate analysis.
- Add comparison report against baseline decoding.
- Add method notes for Medusa, EAGLE, and SpecInfer.
- Add deterministic fixture test for acceptance accounting.

### Milestone 6: Backend Expansion

- Add additional backend adapters if useful.
- Add benchmark result comparison across backends.
- Add documented backend capability matrix.
- Add vLLM benchmark planning workflow for hardware-backed validation.

## 21. Success Criteria

The project is successful when:

- A new user can run a small benchmark from the README.
- Benchmarks produce raw results and a readable report.
- A contributor can add a metric or backend without rewriting the whole codebase.
- The project clearly separates measured data from explanatory notes.
- Tail latency, memory pressure, and batching tradeoffs are first-class outputs.
- The repository is credible as a real open source engineering project, not only a concept document.

## 22. Proposal Completion Criteria

This proposal is complete when it can guide implementation without requiring major product clarification:

- The product scope is explicit.
- The module boundaries are clear.
- The CLI surface is concrete enough to scaffold.
- The configuration model is specified.
- Benchmark outputs and schemas are described.
- Testing and reproducibility expectations are documented.
- Privacy, safety, release, and versioning expectations are documented.
- The open source contribution model is included.
