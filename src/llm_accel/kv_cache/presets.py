from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class KvCachePreset:
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PRESETS: dict[str, KvCachePreset] = {
    "llama-3-8b": KvCachePreset(
        name="llama-3-8b",
        layers=32,
        kv_heads=8,
        head_dim=128,
        description="Llama 3 8B-style grouped-query attention shape.",
    ),
    "llama-3.2-1b": KvCachePreset(
        name="llama-3.2-1b",
        layers=16,
        kv_heads=8,
        head_dim=64,
        description="Llama 3.2 1B-style grouped-query attention shape.",
    ),
    "mistral-7b": KvCachePreset(
        name="mistral-7b",
        layers=32,
        kv_heads=8,
        head_dim=128,
        description="Mistral 7B-style grouped-query attention shape.",
    ),
    "qwen2.5-7b": KvCachePreset(
        name="qwen2.5-7b",
        layers=28,
        kv_heads=4,
        head_dim=128,
        description="Qwen2.5 7B-style grouped-query attention shape.",
    ),
}


def list_kv_cache_presets() -> list[dict[str, object]]:
    return [preset.to_dict() for preset in PRESETS.values()]


def get_kv_cache_preset(name: str) -> KvCachePreset:
    normalized = name.lower()
    if normalized not in PRESETS:
        supported = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown KV cache preset {name!r}; supported: {supported}")
    return PRESETS[normalized]
