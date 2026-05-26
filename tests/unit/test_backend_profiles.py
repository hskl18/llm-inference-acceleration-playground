from llm_accel.serving.profiles import backend_profile


def test_backend_profile_redacts_remote_urls() -> None:
    profile = backend_profile("openai-compatible", base_url="https://example.com/v1")

    assert profile["base_url"] == "redacted"
    assert profile["client"] == "OpenAICompatibleClient"


def test_backend_profile_reports_vllm_requirements() -> None:
    profile = backend_profile("vllm", base_url="http://localhost:8000/v1")

    assert profile["adapter_status"] == "implemented"
    assert profile["required_environment"]
