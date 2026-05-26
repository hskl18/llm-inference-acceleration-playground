from __future__ import annotations

from dataclasses import asdict, dataclass

from llm_accel.kv_cache.presets import get_kv_cache_preset


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "fp16": 2,
    "float16": 2,
    "int8": 1,
    "fp8": 1,
}


@dataclass(frozen=True)
class KvCacheEstimate:
    preset: str | None
    layers: int
    sequence_length: int
    batch_size: int
    kv_heads: int
    head_dim: int
    dtype: str
    dtype_bytes: int
    bytes: int
    mib: float
    gib: float
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_kv_cache(
    *,
    layers: int,
    sequence_length: int,
    batch_size: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
    preset: str | None = None,
) -> KvCacheEstimate:
    for name, value in {
        "layers": layers,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    normalized_dtype = dtype.lower()
    if normalized_dtype not in DTYPE_BYTES:
        supported = ", ".join(sorted(DTYPE_BYTES))
        raise ValueError(f"unsupported dtype {dtype!r}; supported: {supported}")

    dtype_bytes = DTYPE_BYTES[normalized_dtype]
    total_bytes = layers * sequence_length * batch_size * 2 * kv_heads * head_dim * dtype_bytes
    return KvCacheEstimate(
        preset=preset,
        layers=layers,
        sequence_length=sequence_length,
        batch_size=batch_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=normalized_dtype,
        dtype_bytes=dtype_bytes,
        bytes=total_bytes,
        mib=total_bytes / (1024**2),
        gib=total_bytes / (1024**3),
        explanation=(
            "KV cache memory scales linearly with layers, sequence length, batch size, "
            "KV heads, head dimension, and dtype size; the factor of 2 stores keys and values."
        ),
    )


def estimate_kv_cache_from_preset(
    *,
    preset: str,
    sequence_length: int,
    batch_size: int,
    dtype: str,
    layers: int | None = None,
    kv_heads: int | None = None,
    head_dim: int | None = None,
) -> KvCacheEstimate:
    resolved = get_kv_cache_preset(preset)
    return estimate_kv_cache(
        layers=layers if layers is not None else resolved.layers,
        sequence_length=sequence_length,
        batch_size=batch_size,
        kv_heads=kv_heads if kv_heads is not None else resolved.kv_heads,
        head_dim=head_dim if head_dim is not None else resolved.head_dim,
        dtype=dtype,
        preset=resolved.name,
    )
