# Ollama Native Cross-Check

- Model: `qwen2.5:1.5b-instruct`
- Prompts: `8`
- Requested output tokens: `64`
- Workload: `prompt file crosscheck-unique-x2.jsonl`

| Check | Value |
| --- | ---: |
| Output tokens equal to `eval_count` | 1.000 |
| Input tokens equal to `prompt_eval_count` | 1.000 |
| Median decode ms/token relative difference | +0.0088 |
| Median total latency relative difference | -0.3299 |
| Client median ms per output token | 9.48 ms |
| Native `eval_duration / eval_count` median | 9.42 ms |
| Native cold prefill median (`load` + `prompt_eval`) | 263.60 ms |
| Client warm TTFT median | 17.91 ms |
| Median `prompt_eval_count` | 375.5 |
| Median cold `prompt_eval_cached_count` (`/api/chat`) | 34.0 |
| Median warm `prompt_eval_cached_count` (`/api/generate`) | 374.5 |

Per-prompt records are in `cross_check.json`.

## Notes

- Routes run one full pass at a time in a fixed order: /api/chat cold, then /api/generate warm, then the OpenAI client warm.
- Ollama caches served prompts, so only the first route to see a prompt measures a real prefill; the cache_state field on each route records which state it was in.
- Token counts and per-token decode cost do not depend on the cache, so those comparisons are valid across routes.
- Ollama reports eval_duration over all generated tokens, so the client figure compared against it is the decode span divided by output tokens minus one.
- Prefill is reported as absolute medians with their cache state rather than as an agreement ratio between a cold and a warm measurement.
- /api/generate applies the model's chat template with its default system message, so its prompt token count need not equal the /v1 chat count.
- prompt_eval_cached_count shows how much of each prompt the runner served from its resident prefix rather than re-computing.
