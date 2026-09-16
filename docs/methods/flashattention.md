# FlashAttention

FlashAttention (Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135; FlashAttention-2, arXiv:2307.08691) computes exact attention while avoiding the materialization of the full attention matrix in high-bandwidth memory.

## What it changes

Standard attention writes an `n x n` score matrix to HBM, then reads it back for softmax and for the value product.
FlashAttention tiles the computation so that scores stay in on-chip SRAM, using an online softmax that keeps a running maximum and normalizer per tile.
The result is bitwise-comparable exact attention with memory traffic that is linear rather than quadratic in sequence length, and peak activation memory that no longer grows with the square of the sequence.

It is an IO optimization, not an approximation: it does not change what the model outputs, so it cannot be validated by an output-quality check.

## Why it matters for what this tool measures

FlashAttention acts mostly on the prefill phase, where a full prompt is attended to at once.

- **TTFT** is the metric that moves. Prefill is compute-bound and attention-heavy, so a faster attention kernel shortens the time before the first token. Longer `--input-tokens` widens the effect, because the saved traffic grows quadratically with prompt length.
- **TPOT** moves much less. During decode, each step attends one query against the cached keys and values; that step is dominated by reading the KV cache from HBM, not by the attention matrix.
- **Throughput** rises indirectly. Lower activation memory during prefill leaves more of the GPU budget for the KV cache, which raises the number of sequences that fit at a given `--gpu-memory-utilization` and therefore the concurrency the server can sustain.

The right experiment shape in this project is a concurrency and prompt-length sweep with fixed output length, reading TTFT percentiles rather than a single mean:

```bash
llm-accel bench latency --backend vllm --input-tokens 4096 --output-tokens 128 --concurrency 8 ...
```

## Relationship to the serving stack

In vLLM, attention backends are chosen per platform and model, and PagedAttention governs how the KV cache is stored while the attention kernel governs how it is read.
The two are complementary: `llm-accel kv-cache estimate` sizes the cache, while the attention backend determines how fast that cache is consumed.

This project cannot select or measure an attention backend directly.
It records the exact server command, so the attention backend a run used is whatever that command and the server's platform selected.
A FlashAttention claim therefore requires two servers started with different attention backends and the same model, revision, dtype, and hardware, compared through `llm-accel report compare` like any other treatment.
