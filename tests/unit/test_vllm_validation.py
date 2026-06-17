from llm_accel.serving.vllm_validation import validate_vllm_environment


def test_validate_vllm_environment_writes_report(tmp_path) -> None:
    report = validate_vllm_environment(
        model="test-model",
        base_url="mock://local",
        output_dir=tmp_path,
        enable_chunked_prefill=True,
    )

    assert "ready_for_hardware_benchmark" in report
    assert "--enable-chunked-prefill" in report["command"]["argv"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "vllm_validation.json").exists()
    assert (tmp_path / "vllm_validation.md").exists()
