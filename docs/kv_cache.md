# KV Cache

The estimator uses:

```text
layers * sequence_length * batch_size * 2 * kv_heads * head_dim * dtype_bytes
```

The `2` factor accounts for keys and values. Explicit `kv_heads` and `head_dim` parameters let the estimator represent multi-head attention, multi-query attention, and grouped-query attention.

## Explicit Shape

```bash
llm-accel kv-cache estimate \
  --layers 32 \
  --seq-len 8192 \
  --batch-size 16 \
  --kv-heads 8 \
  --head-dim 128 \
  --dtype fp16 \
  --json
```

The JSON output includes raw bytes, MiB, GiB, the resolved model/workload parameters, and a short explanation of the scaling factors.

## Presets

Built-in presets capture common architecture shapes so users do not have to remember layer, KV-head, and head-dimension values for every quick estimate.

```bash
llm-accel kv-cache presets
```

```bash
llm-accel kv-cache estimate \
  --preset llama-3-8b \
  --seq-len 8192 \
  --batch-size 16 \
  --dtype fp16 \
  --json
```

Preset values are convenience defaults for estimation. Check the exact model config before publishing memory claims for a specific checkpoint. Explicit `--layers`, `--kv-heads`, and `--head-dim` values can override the preset when needed.

## Long-Context Example

Long context and concurrency multiply directly. Doubling either `--seq-len` or `--batch-size` doubles the KV cache estimate. This is why a model that fits at short context can fail under a long-context concurrent workload even when weights fit in memory.
