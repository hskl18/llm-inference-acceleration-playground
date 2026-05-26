# Quality Sanity Evaluation

The quality sanity evaluator is intentionally lightweight. It checks fixed prompts for non-empty output, output length, latency metadata, and errors.

```bash
llm-accel eval sanity \
  --base-url mock://local \
  --model mock-model \
  --prompts configs/spec_prompts.jsonl
```

Outputs:

- `manifest.json`
- `quality_eval.json`
- `quality_eval.md`

This does not replace task-specific evaluation. It is a guardrail for benchmark comparisons, especially quantization runs where speed improvements should not hide broken output.

## Task Evaluation

Task evaluation accepts JSONL records with a prompt and expected keywords:

```json
{"prompt": "Explain KV cache and batching.", "expected_keywords": ["kv", "cache", "batching"]}
```

Run it with:

```bash
llm-accel eval task --tasks configs/task_eval_small.jsonl
```

Outputs:

- `manifest.json`
- `task_eval.json`
- `task_eval.md`

Keyword scoring is intentionally simple and inspectable. It is a project guardrail, not a replacement for a domain-specific evaluation suite.
