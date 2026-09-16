# SpecInfer

SpecInfer (Miao et al., "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification", arXiv:2305.09781) generalizes speculative decoding from a single draft chain to a tree of candidate sequences verified in one pass.

## What it changes

Chain speculation proposes one sequence of `g` tokens. One wrong token discards everything after it, which is why the expected yield is the geometric sum `1 + a + ... + a^g`.

SpecInfer proposes a token tree, typically from several small models or several samples of one, merges the proposals into a single tree, and verifies the whole tree in one target forward pass using a tree attention mask that lets each node attend only to its own ancestors.
Verification walks the tree and accepts the longest path the target model's own sampling would have produced, so the output distribution is preserved.

## Why the tree matters

The tree changes the shape of the yield, not just its size.
A chain accepts a prefix; a tree accepts the best path among many prefixes. For the same number of target forward passes, a tree recovers from a single bad token by following a sibling branch instead of stopping.

This is the main reason the closed form in `llm-accel speculative run` is a lower bound for tree-based methods.
That model assumes one chain with a uniform acceptance rate, so it predicts the chain baseline. Medusa, EAGLE-2, and SpecInfer all verify trees and should beat it at the same per-token acceptance rate.

The cost side moves too. Tree width `w` multiplies the verified token count, so the target step is no longer free: attention now runs over `w * g` candidate positions instead of `g`. In the closed form's terms, widening the tree raises effective `c` and the total verification cost, which is why tree width has an optimum rather than growing without limit.

## What this project can and cannot measure

The analytical model here takes a single acceptance rate and a single lookahead. It does not model tree width, tree topology, or multi-draft merging, and it should not be presented as a SpecInfer prediction.

What can be measured on a real server:

- vLLM's speculative counters give accepted tokens, draft tokens, and drafts. `vllm:spec_decode_num_accepted_tokens_per_pos_total` is labelled by position, which is the closest available view of how deep acceptance reaches and therefore of whether extra depth is paying.
- End-to-end TPOT, throughput, and latency percentiles, from a matrix cell against an identically configured baseline.
- Output quality, via `llm-accel eval task` on both endpoints, which is the check that a tree verifier really is distribution-preserving in practice.

vLLM's own tree-shaped methods are configured through the same `--speculative-config` object as any other method, so the tooling and evidence rules in this project apply unchanged: the exact server command is hashed into the optimization profile, and a treatment is only a treatment when the command differs and everything else matches.
