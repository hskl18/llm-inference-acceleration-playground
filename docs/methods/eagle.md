# EAGLE

EAGLE (Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", arXiv:2401.15077; EAGLE-2, arXiv:2406.16858) drafts at the feature level rather than the token level.

## What it changes

A draft model predicts the next token from the tokens so far, which throws away everything the target model already computed.
EAGLE instead runs a single small autoregressive head over the target model's second-to-top hidden states, predicting the *next feature vector*, and only then maps that feature to a token through the target model's own LM head.

The key observation is that a feature sequence is more regular than a token sequence, but is ambiguous without knowing which token was actually sampled. EAGLE resolves that by feeding the already-sampled token of step `t` into the head alongside the feature, which removes the uncertainty that limits feature-level drafting.

EAGLE-2 makes the candidate tree dynamic: draft confidence approximates acceptance probability well, so the tree is expanded where acceptance is likely instead of using a fixed shape.

## How it maps onto the model in this project

In terms of the `llm-accel speculative run` parameters:

- `c` is small, typically well under 0.1. The head is one lightweight decoder layer reusing target features, so `g*c + 1` stays near 1 and long lookaheads stay affordable.
- `a` is high relative to token-level drafting at a comparable cost, which is what makes EAGLE attractive: both terms of the closed form move in the favourable direction at once.
- Because both `a` is high and `c` is low, the predicted speedup is sensitive to `g`. Sweep `--lookahead` rather than fixing it.

As with Medusa, the closed form assumes a chain. EAGLE's tree verification means the analytical number underestimates the achievable speedup, and EAGLE-2's dynamic tree widens that gap further.

EAGLE reuses target hidden states, so the draft head is bound to a specific target model and revision. An EAGLE head paired with a different target checkpoint is not the same treatment.

## Serving and measurement

```bash
vllm serve <target-model> --speculative-config '{"method":"eagle","model":"<eagle-head>","num_speculative_tokens":5}'
```

`llm-accel vllm command --speculative-method eagle --speculative-model <head> --num-speculative-tokens 5` generates that form; `eagle3` is configured the same way.

What this project can establish about an EAGLE deployment:

- The served acceptance rate and the observed draft tokens per draft, from `vllm:spec_decode_num_accepted_tokens_total`, `vllm:spec_decode_num_draft_tokens_total`, and `vllm:spec_decode_num_drafts_total` on `/metrics`.
- The end-to-end TPOT and throughput change, from a matrix cell whose profile declares the speculative model, its immutable revision, and the token count, compared against a baseline cell on an otherwise identical server.
- The output-quality delta, from `llm-accel eval task` against both endpoints.

Speculative decoding mainly shortens TPOT, not TTFT, and it trades extra compute per step for fewer steps. That trade stops paying at high concurrency, where the target model is already compute-saturated, so an EAGLE claim should state the concurrency and request rate it was measured at.
