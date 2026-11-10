# Comparison With Other Tools

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
  The raw offsets also show how many responses the server actually streamed at once, so offered concurrency that only queued is visible.

The honest summary: those tools are better at generating load, this one is stricter about what the resulting number is allowed to claim.
It is also younger.
