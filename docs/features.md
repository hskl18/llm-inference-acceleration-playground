# Feature Reference

This is the complete list of what the toolkit currently does.
The README covers what the tool is for; this page is the inventory.

See [docs/cli.md](cli.md) for the commands, [docs/benchmark_methodology.md](benchmark_methodology.md) for how the measurements are defined, and [docs/result_schemas.md](result_schemas.md) for what each artifact contains.

## Measurement

- OpenAI-compatible streaming and non-streaming client built on the standard library
- `/v1/chat/completions` and `/v1/completions` endpoints
- TTFT measured from the first server-sent event that carries non-empty generated text
- reasoning deltas counted as generated tokens but kept out of the recorded output text
- TPOT from the run's authoritative output token count
- inter-token latency distribution from per-chunk stream arrivals, retained per request
- goodput against declared TTFT, TPOT, and end-to-end SLOs
- p50, p95, and p99 for latency, TTFT, TPOT, inter-token latency, queue delay, and end-to-end latency
- token counts from server usage where the endpoint reports it, with the method recorded per request
- exact tokenizer counts at an immutable revision for vLLM evidence
- forced output length through `ignore_eos` and `min_tokens`, on by default for vLLM
- optional GPU memory telemetry through `nvidia-smi`
- backend version read from the serving process rather than the client host

## Load generation

- closed-loop bounded concurrency
- deterministic fixed-cadence open-loop arrivals from a target request rate
- scheduled-arrival, actual-dispatch, client-queue, and end-to-end request timing
- optional spawned multiprocess load generation
- client saturation detection from queue delay and from the load generator's own CPU cost

## Workloads

- synthetic prompts drawn from a seeded PRNG, unique per request, sharing no prefix beyond chance
- warmup prompts drawn from indices after the measured ones so they cannot warm a measured cache entry
- fixed-prompt files in plain text or JSONL for deliberate prefix-reuse experiments
- recorded prompt-set fingerprint, distinct prompt count, shared-prefix token estimate, and shared-prefix fingerprint
- prompt text sent to the endpoint but never written into result metadata

## Experiments

- config-defined sweeps over input length, output length, and concurrency
- `--repeats` with means and 95% Student t confidence intervals
- `report repeats` for repetitions that must use fresh prompts, such as against a prompt-caching server
- randomized, resumable five-profile optimization matrices with three or more repetitions
- quantization comparison across one already-running endpoint per mode
- analytical speculative decoding speedup from the Leviathan et al. 2023 closed form
- measured speculative acceptance read from a vLLM Prometheus endpoint
- KV cache memory estimator with built-in model-shape presets

## Evidence and auditing

- run manifests listing every generated artifact
- raw JSONL and CSV per-request records, summary JSON, and summary Markdown
- versioned structured optimization profiles with exact command and environment fingerprints
- `report validate` over manifests, schema versions, and required fields for every artifact type
- `report claim-audit` for whether one run can support a hardware claim
- `report ranking-audit` for whether a matrix can support a performance ranking
- strict and explicitly stratified comparison modes with per-profile aggregates
- endpoint URL redaction through a single helper used by every writer
- endpoint fingerprints that bind remote evidence without persisting the URL

## Quality

- fixed-prompt quality sanity checks
- validator-based task evaluation with exact match, regex, Draft 2020-12 JSON Schema, long-context retrieval, and keyword validators
- separate task-specification, raw-output, and generated-summary artifacts
- quality gates bound into matrix cells and required for ranking evidence

## Operations

- `doctor` environment and endpoint health check
- backend capability matrix and adapter profiles
- `vllm command` generation of a current `vllm serve` command line
- `vllm validate` readiness check and `vllm plan` hardware runbook
- packaged example configs for installed CLI users
- SVG latency and sweep plots with no plotting dependency
