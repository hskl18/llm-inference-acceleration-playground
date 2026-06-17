# Research Optimization Plan

Last reviewed: 2026-06-17.

This plan summarizes the next optimization directions for the project after reviewing current inference-serving documentation and recent benchmark literature. The project should continue to separate measured claims from runnable workflows: anything requiring GPU hardware should ship as a validated runbook or benchmark configuration until real hardware results are available.

## Priority 1: Benchmark Prefix Reuse and Long-Prefix Workloads

Why it matters:

- vLLM automatic prefix caching reuses KV cache for requests that share prefixes, improving prefill-heavy workloads such as repeated long-document QA and multi-round chat.
- SGLang's radix-cache design makes prefix reuse a first-class serving optimization.
- Prefix reuse affects TTFT and throughput differently from ordinary synthetic token-length sweeps, so it needs dedicated workloads.

Current coverage:

- Fixed-prompt benchmark files exist.
- Prompt-set fingerprints prevent invalid comparisons across different prompt files.
- `configs/benchmark_prefix_cache.yaml` and `configs/prefix_cache_prompts.jsonl` provide a runnable prefix-reuse smoke workload.
- Fixed-prompt benchmark metadata records `shared_prefix_tokens_estimate` and `shared_prefix_fingerprint` without storing prompt text.

Next implementation:

- Add report text that separates prefill-heavy prefix-reuse observations from decode-heavy throughput results.
- Add larger long-document fixtures for hardware-backed APC/Radix cache validation.

References:

- vLLM Automatic Prefix Caching: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- SGLang RadixAttention paper/project docs: https://docs.sglang.ai/

## Priority 2: Add Benchmark Modes for Server-Side Optimization Flags

Why it matters:

- vLLM exposes server-side controls such as prefix caching, chunked prefill, speculative decoding, quantization, and batching-related limits.
- TensorRT-LLM exposes build/runtime controls around in-flight batching, paged KV cache, KV cache reuse, quantization, and speculative decoding.
- SGLang exposes multiple speculative decoding paths and cache strategies.

Current coverage:

- `llm-accel vllm command` generates a basic vLLM startup command.
- `backend list/profile` now includes common OpenAI-compatible serving engines and optimization metadata.
- vLLM command, validation, and plan artifacts support prefix caching, chunked prefill, batching limits, and speculative decoding startup flags.

Next implementation:

- Add a backend-agnostic `optimization_profile` metadata field to benchmark summaries.
- Add comparison checks so runs with different optimization profiles are not ranked as equivalent.

References:

- vLLM feature docs: https://docs.vllm.ai/en/latest/features/
- TensorRT-LLM workflow and quantization docs: https://nvidia.github.io/TensorRT-LLM/architecture/workflow.html
- SGLang speculative decoding docs: https://docs.sglang.ai/advanced_features/speculative_decoding.html

## Priority 3: Improve Client-Side Benchmark Fidelity at High Concurrency

Why it matters:

- High-concurrency benchmark clients can become the bottleneck and distort TTFT/TPOT measurements.
- The current runner uses a single process with a thread pool, which is acceptable for smoke and small experiments but should not be treated as production-scale load generation.

Current coverage:

- Request failures, timeouts, measured elapsed time, TTFT, TPOT, and throughput are recorded.
- The release gate validates small benchmark workflows without GPU hardware.

Next implementation:

- Add a benchmark warning when concurrency exceeds a conservative single-process threshold.
- Add optional multi-process load generation for high-concurrency endpoint benchmarking.
- Record client process count, worker count, and client-side queueing estimates in run metadata.

Reference:

- Measurement-bias paper: https://arxiv.org/abs/2605.24217

## Priority 4: Expand Quality and Correctness Guardrails

Why it matters:

- Speedups from quantization or speculative decoding should not hide broken output behavior.
- Different serving systems expose structured-output, guided decoding, and speculative modes with different correctness risks.

Current coverage:

- Fixed-prompt sanity evaluation exists.
- Keyword-rubric task evaluation exists.
- Quantization comparison records quality sanity checks.

Next implementation:

- Add JSONL task fixtures for long-context QA, instruction following, and structured-output checks.
- Add optional exact-match / regex / JSON-schema validators.
- Include quality summary deltas in comparison reports.

References:

- SGLang structured-output and speculative-decoding docs: https://docs.sglang.ai/
- vLLM structured-output and speculative-decoding docs: https://docs.vllm.ai/en/latest/features/

## Priority 5: Keep Hardware Claims Strict

Why it matters:

- Quantization support varies by backend, GPU generation, model format, and server version.
- Real benchmark claims need endpoint readiness, artifact validation, and hardware telemetry.

Current coverage:

- `vllm validate` records blockers.
- `report validate` checks generated run directories.
- `docs/release.md` defines benchmark-claim rules.

Next implementation:

- Add a `claim audit` command that checks a run directory for publishable hardware-claim evidence.
- Add backend-version and server-flag capture to every benchmark run when available.
- Add a hardware-backed result template that is explicitly excluded from the default repo until real measurements are collected.

References:

- vLLM quantization hardware support: https://docs.vllm.ai/en/latest/features/quantization/
- TensorRT-LLM benchmarking and performance docs: https://nvidia.github.io/TensorRT-LLM/
