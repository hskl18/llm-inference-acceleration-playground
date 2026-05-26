from llm_accel.kv_cache.estimator import estimate_kv_cache, estimate_kv_cache_from_preset
from llm_accel.kv_cache.presets import get_kv_cache_preset, list_kv_cache_presets


def test_kv_cache_estimate_uses_expected_formula() -> None:
    estimate = estimate_kv_cache(
        layers=2,
        sequence_length=4,
        batch_size=3,
        kv_heads=5,
        head_dim=7,
        dtype="fp16",
    )

    assert estimate.bytes == 2 * 4 * 3 * 2 * 5 * 7 * 2
    assert estimate.mib == estimate.bytes / (1024**2)
    assert estimate.gib == estimate.bytes / (1024**3)
    assert "factor of 2" in estimate.explanation


def test_kv_cache_rejects_unknown_dtype() -> None:
    try:
        estimate_kv_cache(
            layers=1,
            sequence_length=1,
            batch_size=1,
            kv_heads=1,
            head_dim=1,
            dtype="weird",
        )
    except ValueError as exc:
        assert "unsupported dtype" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_kv_cache_preset_resolves_model_shape() -> None:
    preset = get_kv_cache_preset("llama-3-8b")
    estimate = estimate_kv_cache_from_preset(
        preset="llama-3-8b",
        sequence_length=8,
        batch_size=2,
        dtype="fp16",
    )

    assert estimate.preset == "llama-3-8b"
    assert estimate.layers == preset.layers
    assert estimate.kv_heads == preset.kv_heads
    assert estimate.head_dim == preset.head_dim
    assert estimate.bytes == preset.layers * 8 * 2 * 2 * preset.kv_heads * preset.head_dim * 2


def test_kv_cache_preset_can_be_overridden() -> None:
    estimate = estimate_kv_cache_from_preset(
        preset="llama-3-8b",
        sequence_length=8,
        batch_size=2,
        dtype="fp16",
        kv_heads=4,
    )

    assert estimate.kv_heads == 4


def test_list_kv_cache_presets_contains_shapes() -> None:
    presets = list_kv_cache_presets()

    assert any(preset["name"] == "llama-3-8b" for preset in presets)
    assert all("layers" in preset and "kv_heads" in preset and "head_dim" in preset for preset in presets)
