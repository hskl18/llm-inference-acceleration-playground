# Speculative Decoding

`llm-accel speculative run` evaluates the closed-form speedup model from Leviathan, Kalman and Matias, "Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192, ICML 2023).

It is an analytical model, not a serving benchmark. It exists to make the relationship between draft quality, lookahead, and draft cost explicit before anyone spends GPU time.

## The Model

For acceptance rate `a`, lookahead `g` draft tokens proposed per target step, and draft cost ratio `c` (one draft step as a fraction of one target step):

```
expected tokens per target step = (1 - a^(g+1)) / (1 - a)
expected walltime improvement   = (1 - a^(g+1)) / ((1 - a) * (g * c + 1))
```

The numerator is the geometric sum `1 + a + ... + a^g`: one token always comes from the target model's own verification step, and each additional token requires every preceding draft token to be accepted.
The denominator is the cost of one speculative iteration, `g` draft steps plus one target step, measured in target steps.

Limits behave as expected.
At `a = 0` no draft token survives, giving 1 token per target step and a speedup of `1 / (g*c + 1)`, which is strictly a slowdown.
At `a = 1` every draft token survives, giving `g + 1` tokens per target step.
The `a = 1` case is computed as the limit because the closed form is `0/0` there.

Inputs are validated: `0 <= a <= 1`, `g >= 1`, `c >= 0`.

```bash
llm-accel speculative run \
  --acceptance-rate 0.7 \
  --lookahead 4 \
  --draft-cost-ratio 0.2 \
  --output-dir results/runs/speculative-analysis
```

## Measured Acceptance

An assumed acceptance rate is only an assumption. vLLM publishes the real counters on its Prometheus endpoint, so `--metrics-base-url` reads them and uses the served rate instead:

```bash
llm-accel speculative run \
  --metrics-base-url http://localhost:8000/v1 \
  --lookahead 4 \
  --draft-cost-ratio 0.2
```

The counters come from `SpecDecodingProm` in `vllm/v1/spec_decode/metrics.py`:

- `vllm:spec_decode_num_drafts_total`
- `vllm:spec_decode_num_draft_tokens_total`
- `vllm:spec_decode_num_accepted_tokens_total`

The acceptance rate is accepted tokens divided by draft tokens, and draft tokens divided by drafts gives the observed lookahead, which should match the `num_speculative_tokens` the server was started with.
`acceptance_rate_source` records `vllm_metrics` or `declared` so a report never presents an assumption as a measurement.
When the endpoint is unreachable the run still completes from the declared rate and records the error.

## Artifacts

`llm-accel speculative run` writes:

- `speculative_summary.json`
- `speculative_summary.md`
- `acceptance_curve.json`
- `baseline_comparison.json`
- `baseline_comparison.md`

The acceptance curve sweeps the acceptance rate at a fixed lookahead and draft cost ratio, which is the sensitivity that matters when choosing a draft model.

The baseline comparison contrasts target-only decoding, which produces exactly one token per target step at unit cost, with the speculative configuration.

## What the Model Does Not Cover

Verification overhead beyond one target forward pass, batch composition, KV-cache pressure from the draft model, rejection-sampling cost, and kernel efficiency are all outside the model.
A predicted speedup is a ceiling to test against a real serving benchmark, not a result.

The closed form also assumes a single draft chain.
Methods that verify a tree of candidates accept the best path rather than the longest prefix, so for them the model is a lower bound rather than a ceiling.

## Method Notes

- [Medusa](methods/medusa.md): extra decoding heads on the target model, very low draft cost.
- [EAGLE](methods/eagle.md): feature-level drafting, high acceptance at low draft cost.
- [SpecInfer](methods/specinfer.md): tree-based proposal and verification, and why it beats the chain model.
