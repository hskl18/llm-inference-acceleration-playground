from llm_accel.serving.capabilities import get_capability, list_capabilities


def test_capability_matrix_includes_vllm() -> None:
    capabilities = list_capabilities()

    assert "vllm" in capabilities
    assert "awq" in capabilities["vllm"]["quantization_modes"]
    assert "prefix_caching" in capabilities["vllm"]["optimization_features"]


def test_capability_matrix_includes_common_openai_compatible_backends() -> None:
    capabilities = list_capabilities()

    assert "sglang" in capabilities
    assert "radix_cache" in capabilities["sglang"]["optimization_features"]
    assert "tensorrt-llm" in capabilities
    assert "kv_cache_reuse" in capabilities["tensorrt-llm"]["optimization_features"]
    assert "tgi" in capabilities


def test_unknown_backend_returns_unknown_capability() -> None:
    capability = get_capability("custom")

    assert capability["streaming"] == "unknown"
    assert capability["optimization_features"] == ["unknown"]
