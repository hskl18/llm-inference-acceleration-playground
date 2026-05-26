from llm_accel.quantization.sanity import run_quality_sanity_check


def test_quality_sanity_check_passes_for_mock_backend() -> None:
    result = run_quality_sanity_check(
        base_url="mock://local",
        model="mock-model",
        backend="mock",
        quantization="none",
        prompts=["hello"],
    )

    assert result["passed"] is True
    assert result["checks"][0]["non_empty"] is True
