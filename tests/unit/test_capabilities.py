from llm_accel.serving.capabilities import get_capability, list_capabilities


def test_capability_matrix_includes_vllm() -> None:
    capabilities = list_capabilities()

    assert "vllm" in capabilities
    assert "awq" in capabilities["vllm"]["quantization_modes"]


def test_unknown_backend_returns_unknown_capability() -> None:
    capability = get_capability("custom")

    assert capability["streaming"] == "unknown"
