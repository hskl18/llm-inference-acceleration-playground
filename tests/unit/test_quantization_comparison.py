from llm_accel.quantization.comparison import compare_quantization_modes


def test_quantization_comparison_marks_unsupported_modes(tmp_path) -> None:
    report = compare_quantization_modes(
        base_url="mock://local",
        model="mock-model",
        modes=["none", "fp8"],
        output_dir=tmp_path,
        request_count=1,
        backend="mock",
    )

    supported, unsupported = report["runs"]

    assert supported["quantization"] == "none"
    assert supported["support_status"] == "supported"
    assert supported["measured"] is True
    assert unsupported["quantization"] == "fp8"
    assert unsupported["support_status"] == "unsupported"
    assert unsupported["measured"] is False
    assert unsupported["summary_path"] is None
    assert any("not listed as supported" in warning for warning in report["warnings"])
    assert not (tmp_path / "fp8" / "summary.json").exists()
