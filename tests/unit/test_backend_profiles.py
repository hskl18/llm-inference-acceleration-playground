import inspect
from pathlib import Path

import llm_accel
from llm_accel.metrics.execution_identity import displayed_base_url
from llm_accel.serving.profiles import backend_profile


def test_endpoint_redaction_has_exactly_one_implementation() -> None:
    package_root = Path(inspect.getfile(llm_accel)).parent
    inlined = [
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*.py")
        if 'startswith(("mock://", "http://localhost"' in path.read_text(encoding="utf-8")
        or 'startswith(("http://localhost"' in path.read_text(encoding="utf-8")
    ]

    assert inlined == ["metrics/execution_identity.py"]
    assert displayed_base_url("https://vllm.internal.example/v1") == "redacted"
    assert displayed_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert displayed_base_url("mock://local") == "mock://local"


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
