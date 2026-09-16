# Local Apple Silicon Ollama Study, 2026-09-16

This is the first real measured result published in this repository.

It was produced on one laptop against a local Ollama server.
It is not a vLLM result, not a GPU result, and not a claim about any hardware other than the machine described below.
The repository's own claim audit rejects every run in this bundle as publishable hardware evidence, and `claim_audit.json` records that rejection.

## What was measured

| Item | Value |
| --- | --- |
| Server | Ollama 0.34.1, read from the server's `GET /api/version` |
| Endpoint | `http://127.0.0.1:11434/v1`, OpenAI-compatible chat completions, streaming |
| Model | `qwen2.5:1.5b-instruct`, GGUF, `Q4_K_M` |
| Model digest | `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b` |
| Host | Apple M3 Pro, 11 logical cores (5 performance, 6 efficiency), 18 GiB unified memory, macOS 27.0 |
| Accelerator | Apple unified memory; no NVIDIA GPU, so `nvidia-smi` telemetry is absent from every run |
| Benchmark code | commit `140f2543bef53d40d0c3ffb3057db1ea01988249`, recorded in every `run_metadata.json` |
| Workload | 32 requests per run, 64 output tokens, roughly 375 prompt tokens, closed-loop |
| Repetitions | 3 per configuration, each with its own fresh prompt set |
| Token counts | `server_usage`; Ollama returns usage on streamed responses |
| Failed requests | 0 across all 18 runs |

Total measurement time was about 10 minutes for the 18 benchmark runs and about 35 seconds for the cross-check.

## What Ollama actually supports

The study was designed after probing the endpoint rather than assuming it behaves like vLLM.

- `GET /v1/models` lists the model id only, with no revision or digest.
  The immutable digest comes from Ollama's own `GET /api/tags`, and that is what `model_revision` records.
- Streaming responses do carry usage when the request sends `stream_options: {"include_usage": true}`.
  Every run therefore counts tokens from server usage instead of a whitespace estimate.
- `ignore_eos` and `min_tokens` are silently ignored.
  A request that sends both still stops at its natural EOS and returns HTTP 200, so output length cannot be forced.
  The workload works around this instead: the prompts ask for a paragraph of at least eighty words and cap generation at 64 tokens, so every request stops on `length`.
  All 576 measured requests returned exactly 64 output tokens, which the artifacts confirm.
- Ollama emits exactly one token per stream chunk.
  In every measured request the number of inter-token latency samples equals `output_tokens - 1`, so for this server the chunk-arrival gaps this tool records are per-token latencies.
- Ollama does not expose a Prometheus endpoint or NVIDIA telemetry, so serving state and GPU memory are unavailable.

## Result 1: concurrency buys queueing, not throughput

Concurrency sweep, unique prompts, closed-loop, 3 repetitions, means with 95% confidence intervals.

| Concurrency | Output tokens/sec | TTFT p50 (ms) | Inter-token latency p50 (ms) | Latency p95 (ms) | SLO attainment |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65.7 [60.9, 70.5] | 287 [228, 345] | 10.03 [9.75, 10.32] | 1150 [988, 1312] | 0.97 |
| 2 | 62.9 [47.8, 78.1] | 1266 [1059, 1472] | 10.19 [10.01, 10.38] | 2579 [633, 4525] | 0.03 |
| 4 | 62.7 [43.1, 82.3] | 3306 [2304, 4308] | 10.57 [7.20, 13.94] | 4825 [2095, 7555] | 0.02 |
| 8 | 61.1 [44.3, 77.9] | 7602 [4910, 10293] | 10.52 [8.06, 12.97] | 9052 [6658, 11446] | 0.03 |

The SLO was declared before the sweep as TTFT under 1000 ms and TPOT under 15 ms.

Aggregate throughput does not improve with concurrency.
Every confidence interval overlaps every other, and the point estimates drift slightly downward rather than up.
Inter-token latency stays near 10 ms regardless of how many requests are in flight.
Time to first token, in contrast, grows roughly in proportion to concurrency: 287, 1266, 3306, 7602 ms.

That pattern is what a server which admits one request at a time produces.
Serving one request takes about 287 ms of prefill plus 64 tokens at about 10 ms, roughly 930 ms.
A request that arrives behind `k` others therefore waits about `k` times that before its first token, which is what the per-request rows show: at concurrency 4 the four in-flight requests report first-token times of roughly 292, 1210, 2155 and 3071 ms, one service time apart.

Goodput makes the consequence concrete.
At concurrency 1 the server meets the declared budget for 97% of requests.
At every higher concurrency exactly one request in 32 meets it, an attainment of 0.03, while average throughput barely moves.
Reporting only tokens per second would have shown four nearly identical numbers and hidden that three of the four configurations fail the latency budget for almost every user.

This is the opposite of what a continuous-batching server such as vLLM is built to do, and it is the single most useful thing to know before benchmarking anything else on this setup.

### The client was not the bottleneck

Both client-side signals stay flat across the sweep, which is what makes the conclusion about the server safe.

- Client CPU: about 0.010 cores at every concurrency, against a warning threshold of 0.8.
- Client queue delay p95: at most 0.014 ms, against a warning threshold of 10 ms.

## Result 2: a shared prompt prefix cuts time to first token by 6.6x

Both arms use 32 distinct prompts, the same ten source sentences, the same length, concurrency 1, and 3 repetitions with fresh prompts.
They differ only in whether the prompts share a long exact prefix.

| Arm | Prompt tokens | Shared prefix | Output tokens/sec | TTFT p50 (ms) | Inter-token latency p50 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unique prefix | 378 | 4 words | 61.2 [39.6, 82.9] | 317 [268, 365] | 10.35 [7.86, 12.84] |
| Shared prefix | 375 | 293 words | 87.0 [61.4, 112.6] | 47.7 [39.1, 56.3] | 10.38 [7.47, 13.28] |

Time to first token falls 6.6x, from 317 ms to 47.7 ms, and the two confidence intervals are far apart.
End-to-end throughput rises about 42%, from 61.2 to 87.0 tokens per second.
Inter-token latency is unchanged, 10.35 against 10.38 ms, with intervals that overlap almost exactly.

The gain is entirely prefill and none of it is decode, which is what prefix caching should produce and is a useful check that the measurement is measuring what it claims.

The arms are matched rather than merely different.
Both use the same ten handbook sentences and the same 32 questions; the unique arm shuffles the sentences per prompt behind a per-prompt marker word so two prompts diverge within a handful of tokens, while the shared arm keeps one fixed order.
Prompt lengths differ by 3 tokens out of about 376, and each run records `shared_prefix_tokens_estimate` (4 against 293) so the treatment is visible in the artifacts rather than only in this prose.

## Cross-validation against Ollama's own counters

`scripts/ollama_cross_check.py` sends the same prompts through the tool's streaming client and through Ollama's native `/api/chat` and `/api/generate`, then compares what the client timed against what the server reported.

| Check | Unique-prefix prompts | Shared-prefix prompts |
| --- | ---: | ---: |
| Client output tokens equal `eval_count` | 8 of 8 | 8 of 8 |
| Client input tokens equal `prompt_eval_count` | 8 of 8 | 8 of 8 |
| Client ms per output token | 9.48 | 9.47 |
| Server `eval_duration / eval_count` | 9.42 | 9.33 |
| Median relative difference | +0.9% | +2.3% |
| Server cold prefill, `load_duration` + `prompt_eval_duration` | 263.6 ms | 40.5 ms |
| Server `prompt_eval_count` | 375.5 | 373.0 |
| Server `prompt_eval_cached_count`, cold | 34.0 | 355.0 |

Token counts agree exactly, in both directions, on every prompt.
That matters because the tool's token throughput and TPOT are computed from those counts.

Decode cost agrees to within 0.9% and 2.3%, with the client always slightly higher.
That direction is expected: the client's figure includes server-sent-event framing and parsing that `eval_duration` does not, and 0.06 to 0.14 ms per token is a plausible size for it.

Prefill agrees at the arm level.
The server's cold prefill plus one token of decode is about 273 ms for unique prompts and about 50 ms for shared ones, against benchmark TTFT p50 of 317 ms and 47.7 ms.
The shared arm matches within 4%.
The unique arm's benchmark figure is about 16% above the cross-check figure, which is consistent with the higher background load during the benchmark phase described below; it is a discrepancy worth naming rather than smoothing over.

`prompt_eval_cached_count` independently confirms the prefix mechanism.
With unique prompts the server re-computes almost the whole prompt, caching only 34 of 375 tokens, which is the chat template header.
With shared prompts it serves 355 of 373 tokens from cache and re-computes only the differing tail.

### A confound this study had to fix twice

Ollama keeps served prompts in its runner cache, and that broke two earlier versions of this study before the published one.

The first version used `--repeats 3` over one fixed prompt file.
At concurrency 1 the first repetition measured 264 ms TTFT and the next two measured 17 ms, because repetitions two and three were reading the cache that repetition one had filled.
Averaging those three numbers would have produced a figure describing nothing.
The fix was to give every repetition its own prompt set, which one rerun of a single configuration cannot do; `llm-accel report repeats` exists because of this measurement.

The second version reused the same prompt run ids on a later execution, which handed the server prompts it still had cached from hours earlier.
The fix was an execution tag in every prompt run id, and `build_prompts.py` now says so in its help text.

The cross-check had the same bug in a different shape.
Its first draft sent each prompt to three routes back to back, so the two later routes measured the cache the first route had warmed; the unique-prefix arm reported 368 of 369 prompt tokens cached and a meaningless prefill comparison.
It now runs one route per pass in a fixed order and labels each route's cache state, because only token counts and per-token decode cost survive a cache hit unchanged.

The general lesson is that the prompt-uniqueness rule this project already applies within a run has to extend across runs whenever the server caches prompts.
A benchmark can be perfectly instrumented and still measure a cache.

## What the audits say about this evidence

`llm-accel report validate` passes on all 26 generated directories in this bundle; `validation.json` records the result.

`llm-accel report claim-audit` rejects it, which is the correct outcome, and `claim_audit.json` records the 15 blockers.
Among them:

- `hardware claims require a vLLM endpoint run, not mock or relabeled output`
- `GPU memory telemetry is unavailable`
- `missing GPU name`, `missing GPU driver version`, `missing CUDA version`, `missing PyTorch version`
- `server_command.txt is required in the run directory`, because Ollama was not launched from a command this tool generated and fingerprinted
- `only 32 measured requests; at least 100 are required`
- `an exact dtype must be recorded`, because Ollama reports a GGUF quantization level rather than an activation dtype

Nothing in this bundle should be quoted as a vLLM number, a GPU number, or a hardware performance claim.
It is a correct, reproducible measurement of one small quantized model on one laptop.

## Measurement conditions and limitations

- The host was not idle.
  `environment_before.json` and `environment_after.json` record load averages between 10.6 and 19.5 across the measurement window, from the machine's other applications.
  This widens the confidence intervals, most visibly at concurrency 4 and 8, and it is the honest explanation for the residual spread rather than a property of the server.
  All six arms were interleaved inside each repetition precisely so that background drift moves every arm together instead of landing on whichever arm happened to run during a busy minute.
- 32 requests per run is a small sample for a p95.
  Treat p50 and the confidence interval over repetitions as the reliable figures here and p95 as indicative.
- Scheduling is closed-loop, which every summary warns is susceptible to coordinated omission.
  A closed-loop run cannot be used for a cross-configuration performance ranking, and the ranking audit would refuse it.
- Warmup is zero on purpose.
  Warmup requests are drawn from the same fixed prompt file, so on a prompt-caching server they would pre-warm the very prompts about to be measured.
- Ollama's parallelism was left at its default and not tuned.
  The serialized behaviour reported here is the behaviour of a default local install, not a statement about Ollama's best achievable throughput.
- One model, one quantization, one prompt shape, one machine.
  Nothing here generalizes to other models, other backends, or other hardware.

## Reproducing this study

Prompt sets are generated rather than stored, because 18 sets of 32 prompts is about a megabyte of near-duplicate text.
`build_prompts.py` is deterministic, and every run records a `workload_fingerprint`, so regenerating with the same arguments and comparing fingerprints proves which prompt set a published run measured.

Pick an execution tag that has never been used against your server, then for each run:

```bash
STUDY=results/published/2026-09-16-ollama-apple-m3-pro
TAG=x2   # the published runs used x2; choose an unused tag for a fresh execution

python $STUDY/build_prompts.py \
  --arm unique --run-id "sweep-c1-r01-$TAG" --count 32 --output /tmp/prompts.jsonl

llm-accel bench throughput \
  --base-url http://127.0.0.1:11434/v1 \
  --backend ollama \
  --model qwen2.5:1.5b-instruct \
  --model-revision 65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b \
  --quantization Q4_K_M \
  --hardware-label apple-m3-pro-18gb \
  --optimization-profile ollama-default \
  --concurrency 1 \
  --output-tokens 64 \
  --request-count 32 \
  --warmup-count 0 \
  --prompts /tmp/prompts.jsonl \
  --slo-ttft-ms 1000 \
  --slo-tpot-ms 15 \
  --output-dir $STUDY/concurrency-sweep/c1/repeat-01
```

Repeat that for concurrencies 1, 2, 4 and 8 and repetitions 01, 02 and 03 with the matching run ids, and for the prefix arms with `--arm shared` and `--arm unique` at concurrency 1.
Interleave the arms within each repetition rather than running each arm to completion in turn.

Aggregate each group and cross-validate:

```bash
llm-accel report repeats \
  --run-dir $STUDY/concurrency-sweep/c1/repeat-01 \
  --run-dir $STUDY/concurrency-sweep/c1/repeat-02 \
  --run-dir $STUDY/concurrency-sweep/c1/repeat-03 \
  --output-dir $STUDY/concurrency-sweep/c1

python scripts/ollama_cross_check.py \
  --model qwen2.5:1.5b-instruct \
  --prompts /tmp/crosscheck.jsonl \
  --max-tokens 64 \
  --output-dir $STUDY/cross-check/unique-prefix

llm-accel report validate --run-dir $STUDY/concurrency-sweep/c1/repeat-01
llm-accel report claim-audit --run-dir $STUDY/concurrency-sweep/c1/repeat-01
```

`report claim-audit` is expected to exit 1.
A benchmark tool that called this a hardware result would be the thing worth distrusting.

## Files in this bundle

```text
build_prompts.py                                   deterministic prompt-set generator
capture_environment.py                             host, load average, and server state capture
environment_before.json, environment_after.json
concurrency-sweep/c{1,2,4,8}/repeat-{01,02,03}/    full run artifacts
concurrency-sweep/c{1,2,4,8}/repeats_summary.json  aggregated means and 95% intervals
prefix-sharing/{unique,shared}-prefix/repeat-{01,02,03}/
prefix-sharing/{unique,shared}-prefix/repeats_summary.json
cross-check/{unique,shared}-prefix/cross_check.json
validation.json                                    report validate over every directory
claim_audit.json                                   the audit's refusal, with all 15 blockers
```

Each run directory holds `raw_requests.jsonl` with one row per request, including its inter-token latency samples, so every published number can be recomputed from raw evidence.
