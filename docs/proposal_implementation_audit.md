# Proposal Implementation Audit

This document maps the proposal roadmap to the current implementation. It is meant for maintainers and contributors who need to understand which product surfaces are runnable, which checks protect benchmark claims, and which work still requires hardware-backed validation.

## Scope Status

| Proposal area | Current implementation | Verification |
| --- | --- | --- |
| Project foundation | Python package under `src/llm_accel`, `llm-accel` console script, MIT license, contribution, security, conduct, release docs | `python3 scripts/release_check.py` |
| Baseline benchmarking | OpenAI-compatible chat/completion client, streaming TTFT capture, deterministic mock backend, latency benchmark, raw JSONL/CSV, summary JSON/Markdown, SVG latency plot | `tests/integration/test_cli.py`, `scripts/smoke.py` |
| Throughput benchmarking | `bench throughput` uses the same request execution path and writes throughput-specific JSON/Markdown artifacts | `test_cli_throughput_benchmark_writes_throughput_summary` |
| Sweeps and reports | YAML/JSON config sweeps, aggregate summaries, plots, report regeneration, validation, comparison reports | `test_cli_sweep_writes_aggregate`, `test_report_validation.py`, `test_report_comparison.py` |
| Fixed-prompt workloads | Plain-text and JSONL prompt loading, prompt-set fingerprinting, shared-prefix metadata, config-relative prompt paths, prompt text excluded from benchmark metadata | `test_cli_latency_benchmark_accepts_fixed_prompt_file`, `test_cli_sweep_accepts_prompt_workload`, `test_shared_prefix_metadata_does_not_expose_prompt_text` |
| KV cache analysis | Formula-based estimator, model-shape presets, JSON output for automation, explanatory docs | `test_kv_cache.py`, `test_cli_kv_cache_json` |
| Quantization comparison | Mode support labeling, unsupported-mode reporting, per-mode benchmark metadata, fixed-prompt sanity checks | `test_quantization_comparison.py`, `test_quantization_sanity.py` |
| Speculative decoding study | Deterministic toy accounting, acceptance curves, baseline comparison artifacts, method notes | `test_speculative_analysis.py` |
| vLLM workflow | Server command helper with optimization flags, readiness validation, backend profiles, hardware benchmark runbook with latency, throughput, sweep, task-eval, and validation steps | `test_vllm.py`, `test_vllm_validation.py`, `test_vllm_plan.py` |
| Backend expansion | Capability/profile metadata for vLLM, SGLang, TensorRT-LLM, TGI, and generic OpenAI-compatible endpoints | `test_capabilities.py`, `test_backend_profiles.py` |
| Installed-package usability | Packaged example configs and prompt/task files exportable with `llm-accel examples write` | `test_cli_examples_write_creates_runnable_configs`, `scripts/smoke.py` |
| Open source hygiene | README, docs, changelog, release gate, issue-label guidance, no committed benchmark claims | `docs/release.md`, `CONTRIBUTING.md` |
| Research-backed optimization plan | Prioritized plan for prefix reuse workloads, optimization-profile metadata, high-concurrency fidelity, quality guardrails, and claim auditing | `docs/research_optimization_plan.md` |

## Product Guarantees

- Benchmark commands write machine-readable artifacts before Markdown and plot generation so raw evidence is preserved if report generation fails.
- Failed requests and timeouts are recorded in raw rows and included in summaries.
- Remote endpoint URLs and secret-like config values are redacted from persisted configs and metadata.
- Comparison reports disable ranking when runs are too small, failed, or incompatible by model, backend, dtype, quantization, API kind, workload mode, or prompt-set fingerprint.
- Mock runs are suitable for workflow validation only and are not hardware performance claims.

## Remaining Hardware Validation

The current repository can fully validate the package, CLI, schemas, mock benchmarks, docs, and smoke workflows without a GPU. Real vLLM performance claims still require a machine where:

- the `vllm` Python package is installed,
- GPU telemetry is available,
- an OpenAI-compatible vLLM endpoint is running,
- `llm-accel vllm validate --smoke` reports no blockers,
- generated benchmark directories pass `llm-accel report validate`.

Until those conditions are met, the project should publish workflow examples and schema fixtures only, not hardware benchmark claims.
