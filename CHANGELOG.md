# Changelog

All notable changes to this project will be documented here.

The project follows semantic versioning after the CLI and result schemas stabilize. During `0.x`, CLI flags and schema details may change with clear notes.

## 0.1.0 - Unreleased

Initial implementation slice:

- Python package and `llm-accel` CLI
- OpenAI-compatible mock, streaming, and non-streaming client paths
- benchmark selection for chat-completion and completion API endpoints
- concurrent latency/throughput benchmark runner
- config-driven sweep runner
- raw request JSONL, summary JSON/Markdown, manifest, and resolved config artifacts
- raw request CSV artifacts for spreadsheet-friendly inspection
- timestamped default benchmark run directories under `results/runs/`
- dependency-free SVG plots
- KV cache estimator
- KV cache model-shape presets and preset-aware CLI estimates
- quantization comparison workflow with quality sanity checks
- quantization support-status reporting for unsupported or unknown modes
- lightweight quality and keyword-rubric task evaluation
- speculative decoding toy accounting and acceptance curves
- speculative decoding baseline comparison artifacts
- backend capability/profile commands
- vLLM command helper and readiness validator
- vLLM hardware benchmark plan generator
- packaged example configs exportable through `llm-accel examples write`
- config-relative fixed-prompt paths for self-contained exported examples
- proposal implementation audit for maintainers and contributors
- research-backed optimization roadmap for prefix reuse, backend flags, benchmark fidelity, and claim auditing
- backend capability metadata for SGLang, TensorRT-LLM, TGI, and common optimization features
- CI matrix coverage for Python 3.13
- run validation and comparison reports
- comparison warnings for low-sample, failed, or incompatible runs
- ruff lint configuration and CI lint gate
- analysis-only report regeneration from existing run directories
- reproducibility metadata for Python, OS, git commit, hardware label, and GPU name
- backend version metadata when available
- per-run warning lists for missing or unsupported measurements
- explicit timeout counts in benchmark summaries
- sweep config validation with inline-secret rejection
- code of conduct and security reporting guidance
- unit and integration test suite
