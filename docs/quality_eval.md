# Quality Evaluation

The quality sanity evaluator checks fixed prompts for non-empty output, output length, latency metadata, and request errors.
It writes verbatim model responses to a raw JSONL artifact instead of embedding generated text in the summary.

```bash
llm-accel eval sanity \
  --base-url mock://local \
  --model mock-model \
  --prompts configs/spec_prompts.jsonl
```

Outputs:

- `manifest.json`
- `quality_outputs.jsonl`
- `quality_eval.json`
- `quality_eval.md`

`quality_eval.json` records a prompt-set SHA-256 fingerprint but does not include prompt text or generated text.
`quality_outputs.jsonl` contains each verbatim generated response and its timing metadata.
Review raw output artifacts for sensitive data before sharing them.

## Task Evaluation

Task evaluation accepts JSONL records with a stable ID, prompt, and validator.
The supported validators are `keywords`, `exact_match`, `regex`, `json_schema`, and `long_context`.

Exact match can explicitly control surrounding whitespace and case sensitivity:

```json
{"id":"exact-answer","prompt":"Return only 42.","validator":{"type":"exact_match","expected":"42","strip":true,"case_sensitive":true}}
```

Regex validation supports `fullmatch` and `search` modes plus `IGNORECASE`, `MULTILINE`, and `DOTALL` flags:

```json
{"id":"invoice-id","prompt":"Return one invoice ID.","validator":{"type":"regex","pattern":"INV-[0-9]{4}","mode":"fullmatch","flags":[]}}
```

JSON Schema validation parses the entire stripped output as JSON and validates it with Draft 2020-12:

```json
{"id":"structured-answer","prompt":"Return a JSON object with an integer answer.","validator":{"type":"json_schema","schema":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["answer"],"properties":{"answer":{"type":"integer"}},"additionalProperties":false}}}
```

Long-context validation requires each expected answer to occur late enough in the input prompt and requires the generated output to contain every answer:

```json
{"id":"late-context-answer","prompt":"LONG_CONTEXT_WITH_ORCHID-731_NEAR_THE_END","validator":{"type":"long_context","expected":["ORCHID-731"],"min_prompt_chars":4096,"min_expected_position":0.75,"case_sensitive":false}}
```

`min_expected_position` is a fraction from `0.0` through `1.0`.
The task loader rejects fixtures whose expected long-context answer does not occur at or after that position.

Legacy records with `expected_keywords` remain supported and are normalized to the `keywords` validator.

```json
{"prompt":"Explain KV cache and batching.","expected_keywords":["kv","cache","batching"]}
```

Run a task evaluation with:

```bash
llm-accel eval task --tasks configs/task_eval_small.jsonl
```

Outputs:

- `manifest.json`
- `task_specs.jsonl`
- `task_outputs.jsonl`
- `task_eval.json`
- `task_eval.md`

`task_specs.jsonl` stores normalized prompts and validators.
`task_outputs.jsonl` stores verbatim generated responses and request metadata.
`task_eval.json` stores aggregate counts, scores, task-set fingerprint, and compact per-case results.
The generated summary intentionally excludes prompts, expected values, regex patterns, JSON schemas, and generated text.
