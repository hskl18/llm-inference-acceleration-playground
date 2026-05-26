from pathlib import Path

import pytest

from llm_accel.config.loader import ConfigError, get_path, load_config, sanitize_resolved_config, validate_benchmark_config


def test_load_config_reads_sample_yaml() -> None:
    config = load_config(Path("configs/benchmark_small.yaml"))

    assert get_path(config, "endpoint.base_url") == "mock://local"
    assert get_path(config, "endpoint.api_kind") == "chat"
    assert get_path(config, "workload.concurrency") == [1, 4]
    assert get_path(config, "run.measured_requests") == 4
    validate_benchmark_config(config)


def test_validate_benchmark_config_rejects_invalid_values() -> None:
    config = {
        "run": {"measured_requests": 0, "warmup_requests": -1, "timeout_seconds": 0},
        "endpoint": {"base_url": "", "api_kind": "responses", "api_key": "sk-test"},
        "model": {"name": ""},
        "workload": {"input_tokens": [128, 0], "output_tokens": [], "concurrency": ["four"]},
    }

    with pytest.raises(ConfigError) as exc_info:
        validate_benchmark_config(config)

    message = str(exc_info.value)
    assert "run.measured_requests must be a positive integer" in message
    assert "endpoint.api_key must not contain an inline secret" in message
    assert "endpoint.api_kind must be 'chat' or 'completion'" in message
    assert "workload.concurrency must be a positive integer" in message


def test_validate_benchmark_config_accepts_prompt_file_without_input_tokens() -> None:
    config = {
        "run": {"measured_requests": 2},
        "endpoint": {"base_url": "mock://local"},
        "model": {"name": "mock-model"},
        "workload": {"prompts_path": "configs/spec_prompts.jsonl", "output_tokens": [8], "concurrency": [1]},
    }

    validate_benchmark_config(config)


def test_sanitize_resolved_config_redacts_remote_endpoint_and_secret_like_keys() -> None:
    config = {
        "endpoint": {
            "base_url": "https://api.example.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "authorization": "Bearer secret",
        },
        "nested": {
            "service_token": "secret",
            "items": [{"secret": "value"}],
        },
    }

    sanitized = sanitize_resolved_config(config)

    assert sanitized["endpoint"]["base_url"] == "redacted"
    assert sanitized["endpoint"]["api_key_env"] == "OPENAI_API_KEY"
    assert sanitized["endpoint"]["authorization"] == "redacted"
    assert sanitized["nested"]["service_token"] == "redacted"
    assert sanitized["nested"]["items"][0]["secret"] == "redacted"
    assert config["endpoint"]["base_url"] == "https://api.example.com/v1"


def test_sanitize_resolved_config_keeps_local_endpoint() -> None:
    config = {"endpoint": {"base_url": "http://localhost:8000/v1"}}

    sanitized = sanitize_resolved_config(config)

    assert sanitized["endpoint"]["base_url"] == "http://localhost:8000/v1"
