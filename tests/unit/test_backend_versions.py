from llm_accel.serving.versions import detect_backend_version


def test_detect_backend_version_reports_mock_version() -> None:
    assert detect_backend_version("mock").startswith("llm-accel-mock/")


def test_detect_backend_version_returns_none_when_unavailable() -> None:
    assert detect_backend_version("openai-compatible") is None
