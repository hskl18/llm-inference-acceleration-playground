# Medusa

Medusa (Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", arXiv:2401.10774) attaches several lightweight decoding heads to the target model itself, so position `t+1`, `t+2`, ... are proposed in the same forward pass that produces position `t`.

## What it changes

A draft-model setup runs a second, smaller model `g` times per target step.
Medusa instead adds `k` extra feed-forward heads on the target model's final hidden state. Head `i` predicts the token `i+1` positions ahead. Because the heads share the backbone activation, the proposal cost is a few extra matrix multiplies, not `g` extra model calls.

Medusa proposes a tree of candidates rather than one chain: each head emits its top-`k` tokens, and the Cartesian product is verified in one attention pass with an attention mask that keeps the branches independent.

## How it maps onto the model in this project

`llm-accel speculative run` takes acceptance rate `a`, lookahead `g`, and draft cost ratio `c`.

- `c` is very small for Medusa, often on the order of a few percent, because the heads reuse the backbone forward pass. The term `g*c + 1` in the denominator stays close to 1, so almost all of the predicted speedup survives.
- `a` is typically lower than for a well-matched draft model, because a single feed-forward head has far less context than a small transformer. Head `i` also degrades as `i` grows.
- The single-`a` closed form is therefore a simplification for Medusa. Per-position acceptance decays with distance, so treating the heads as one rate slightly overstates long lookaheads. Sweeping `--acceptance-rate` at the lookahead under consideration bounds the answer from both sides.

The closed form also assumes chain verification. Tree verification accepts the best path through the candidate tree, so measured acceptance exceeds the chain model's prediction for the same per-head accuracy; treat an analytical Medusa number as a lower bound.

## Serving and measurement

vLLM exposes Medusa through the same interface as every other speculative method:

```bash
vllm serve <target-model> --speculative-config '{"method":"medusa","model":"<medusa-heads>","num_speculative_tokens":4}'
```

Generate this with `llm-accel vllm command --speculative-method medusa --speculative-model <heads> --num-speculative-tokens 4`.

The served acceptance rate is observable: `vllm:spec_decode_num_accepted_tokens_total` divided by `vllm:spec_decode_num_draft_tokens_total` on the server's `/metrics` endpoint, which `llm-accel speculative run --metrics-base-url` reads.

Medusa heads are trained against a specific backbone, so a Medusa run and its baseline must use the same target model and revision. The heads are extra weights and belong in the optimization profile as the speculative model with its own immutable revision.

Output distribution is preserved only when the verification step uses the target model's own acceptance criterion. Run `llm-accel eval task` against both endpoints and report the score delta with any latency claim.
