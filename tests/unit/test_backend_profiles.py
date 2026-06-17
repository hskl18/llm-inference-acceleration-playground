from llm_accel.serving.profiles import backend_profile


def test_backend_profile_redacts_remote_urls() -> None:
    profile = backend_profile("openai-compatible", base_url="https://example.com/v1")

    assert profile["base_url"] == "redacted"
    assert profile["client"] == "OpenAICompatibleClient"


def test_backend_profile_reports_vllm_requirements() -> None:
    profile = backend_profile("vllm", base_url="http://localhost:8000/v1")

    assert profile["adapter_status"] == "implemented"
    assert profile["required_environment"]


def test_backend_profile_reports_sglang_as_openai_compatible() -> None:
    profile = backend_profile("sglang", base_url="http://localhost:30000/v1")

    assert profile["adapter_status"] == "implemented"
    assert "SGLang" in profile["required_environment"][0]
    assert "radix_cache" in profile["capability"]["optimization_features"]
