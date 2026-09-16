# LLM Inference Acceleration Playground

Benchmark an LLM serving endpoint, then prove the number is worth believing.

The tool drives load at any OpenAI-compatible endpoint and records TTFT, inter-token latency, TPOT, throughput, and goodput against a declared SLO.
What it adds beyond that is an audit trail: every published metric is recomputable from the committed per-request rows, and the same tool refuses to call a run a hardware claim when the evidence does not support one.

## A real result

On an Apple M3 Pro laptop, Ollama 0.34.1 serving `qwen2.5:1.5b-instruct`, 3 repetitions per point, 95% confidence intervals:

| Concurrency | Output tokens/sec | TTFT p50 | Requests meeting a 1000 ms TTFT and 15 ms TPOT budget |
| ---: | ---: | ---: | ---: |
| 1 | 65.7 [60.9, 70.5] | 287 ms | 97% |
| 2 | 62.9 [47.8, 78.1] | 1266 ms | 3% |
| 4 | 62.7 [43.1, 82.3] | 3306 ms | 2% |
| 8 | 61.1 [44.3, 77.9] | 7602 ms | 3% |

Throughput is flat, so a report that only published tokens per second would have shown four nearly identical numbers.
Goodput shows what actually happened: this server admits one request at a time, so concurrency buys queueing, and three of the four configurations miss the latency budget for almost every request.

In the same study a shared 293-word prompt prefix cut TTFT p50 by 6.6x, from 317 ms to 47.7 ms, while inter-token latency stayed unchanged at 10.4 ms, which is the signature of a prefill saving rather than a faster model.
Token counts agreed exactly with Ollama's own `eval_count` and `prompt_eval_count` on every cross-checked prompt, and per-token decode cost agreed to within 0.9% and 2.3%.

Full numbers, method, confounds, and limitations: [results/published/2026-09-16-ollama-apple-m3-pro/report.md](results/published/2026-09-16-ollama-apple-m3-pro/report.md).

This is a local Apple Silicon Ollama measurement.
It is not a vLLM result and not a GPU result, and the repository's own claim audit rejects it as hardware evidence with 15 blockers, which is the correct outcome.

## How this differs from the established tools

[`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve.html), [NVIDIA GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md), and [GuideLLM](https://github.com/vllm-project/guidellm) are mature load generators, and all three shape load better than this project does.
`vllm bench serve` has Poisson and gamma arrivals, burstiness control, ramp-up strategies, and a wide set of datasets.
GuideLLM has six load profiles including a sweep that searches for the safe operating range, plus HTML reports.
GenAI-Perf covers Triton and KServe as well as OpenAI-compatible endpoints, though NVIDIA is phasing it out in favour of AIPerf.
If the job is to drive load at a vLLM server and read the numbers, use `vllm bench serve`.

This project starts from a different question: not "what number did we get" but "would this number survive someone checking it".
That leads to work the load generators do not do:

- **The tool audits its own output.** `report claim-audit` refuses to call a run publishable hardware evidence unless the GPU telemetry, the exact serving-command hash, immutable model and tokenizer revisions, server-reported token counts, and the request and warmup minimums are all present. `report ranking-audit` refuses a cross-configuration ranking without three valid repetitions per profile, matching invariants, a passing quality gate, open-loop dispatch evidence, and a client that was not saturated.
- **Published metrics are recomputed from raw rows.** The audit rebuilds every percentile, throughput figure, inter-token latency distribution, and goodput number from `raw_requests.jsonl` and blocks the run if the summary disagrees.
- **Comparisons are gated on invariants rather than assumed.** Optimization settings are treatment dimensions; model, tokenizer, prompt set, schedule, client configuration, quality gate, and environment are invariants, and runs that differ in an invariant are put in separate strata instead of being ranked against each other.
- **Workload identity is recorded.** Every run stores a prompt-set fingerprint, the distinct prompt count, and a shared-prefix estimate, and warns when the measured prompts repeat, because a repeating workload against a caching server measures the cache.
- **The client is treated as a suspect.** Every run publishes the load generator's own CPU cost and its queue delay, so "the server was slow" can be distinguished from "our benchmark could not keep up".

The honest summary: those tools are better at generating load, this one is stricter about what the resulting number is allowed to claim.
It is also younger, with one published real result and no GPU measurements yet.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Quickstart

No GPU is required. The `mock://local` backend is deterministic and exercises the whole pipeline.

```bash
llm-accel doctor

llm-accel bench throughput \
  --base-url mock://local \
  --model mock-model \
  --concurrency 4 \
  --output-tokens 64 \
  --request-count 32 \
  --slo-ttft-ms 1000 \
  --slo-tpot-ms 15 \
  --repeats 3 \
  --output-dir results/runs/quickstart

llm-accel report validate --run-dir results/runs/quickstart/repeat-01
llm-accel report claim-audit --run-dir results/runs/quickstart/repeat-01
```

The claim audit exits non-zero, on purpose: a mock run is not hardware evidence, and the tool says so rather than letting the number escape.

Against a real endpoint, name the backend and the endpoint instead:

```bash
llm-accel bench throughput \
  --base-url http://localhost:8000/v1 \
  --backend vllm \
  --model MODEL_ID \
  --tokenizer MODEL_ID \
  --tokenizer-revision TOKENIZER_REVISION \
  --request-schedule open-loop \
  --request-rate-rps 20 \
  --concurrency 8 \
  --output-dir results/runs/vllm-run
```

## Documentation

- [Command reference](docs/cli.md)
- [Feature list](docs/features.md)
- [Benchmark methodology](docs/benchmark_methodology.md)
- [Result schemas](docs/result_schemas.md)
- [Backend capabilities](docs/backend_capabilities.md)
- [vLLM integration](docs/vllm.md) and the [hardware benchmark runbook](docs/hardware_benchmark_runbook.md)
- [KV cache](docs/kv_cache.md), [quantization](docs/quantization.md), [speculative decoding](docs/speculative_decoding.md), [quality evaluation](docs/quality_eval.md)
- [Release process](docs/release.md) and [proposal coverage map](docs/proposal_implementation_audit.md)

## Development

```bash
python3 scripts/release_check.py --metadata-only
python3 -m ruff check .
python3 -m pytest
python3 scripts/smoke.py
```

The test suite does not require a GPU.

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).
[proposal.md](proposal.md) holds the full product proposal and roadmap.
