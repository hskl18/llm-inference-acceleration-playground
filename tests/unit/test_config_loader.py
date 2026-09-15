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


def test_load_config_uses_yaml_semantics_for_none_and_inline_comments(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  name: m\n  quantization: none\n  dtype: fp16 # half precision\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert get_path(config, "model.quantization") == "none"
    assert get_path(config, "model.dtype") == "fp16"


def test_load_config_does_not_fall_back_to_a_different_parser(tmp_path: Path, monkeypatch) -> None:
    import llm_accel.config.loader as loader

    assert not hasattr(loader, "_load_minimal_yaml")
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("model: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_json(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(config_path)


def test_validate_benchmark_config_rejects_null_quantization_and_dtype() -> None:
    config = {
        "run": {"measured_requests": 2},
        "endpoint": {"base_url": "mock://local"},
        "model": {"name": "mock-model", "quantization": None, "dtype": None},
        "workload": {"input_tokens": [8], "output_tokens": [8], "concurrency": [1]},
    }

    with pytest.raises(ConfigError) as exc_info:
        validate_benchmark_config(config)

    assert "model.quantization must be a non-empty string" in str(exc_info.value)
    assert "model.dtype must be a non-empty string" in str(exc_info.value)


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


def test_validate_benchmark_config_accepts_open_loop_client_settings() -> None:
    config = {
        "run": {
            "measured_requests": 4,
            "client_processes": 2,
            "queue_delay_warning_ms": 5.0,
        },
        "endpoint": {"base_url": "mock://local"},
        "model": {"name": "mock-model"},
        "workload": {
            "input_tokens": [16],
            "output_tokens": [8],
            "concurrency": [2, 4],
            "request_schedule": "open-loop",
            "request_rate_rps": 100.0,
        },
    }

    validate_benchmark_config(config)


def test_validate_benchmark_config_rejects_invalid_client_schedule() -> None:
    config = {
        "run": {"measured_requests": 4, "client_processes": 2},
        "endpoint": {"base_url": "mock://local"},
        "model": {"name": "mock-model"},
        "workload": {
            "input_tokens": [16],
            "output_tokens": [8],
            "concurrency": [1],
            "request_schedule": "open-loop",
        },
    }

    with pytest.raises(ConfigError) as exc_info:
        validate_benchmark_config(config)

    message = str(exc_info.value)
    assert "workload.request_rate_rps is required" in message
    assert "run.client_processes must not exceed" in message


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_validate_benchmark_config_rejects_nonfinite_client_values(invalid: float) -> None:
    config = {
        "run": {
            "measured_requests": 4,
            "timeout_seconds": invalid,
            "queue_delay_warning_ms": invalid,
        },
        "endpoint": {"base_url": "mock://local"},
        "model": {"name": "mock-model"},
        "workload": {
            "input_tokens": [16],
            "output_tokens": [8],
            "concurrency": [1],
            "request_schedule": "open-loop",
            "request_rate_rps": invalid,
        },
    }

    with pytest.raises(ConfigError) as exc_info:
        validate_benchmark_config(config)

    message = str(exc_info.value)
    assert "run.timeout_seconds must be a positive number" in message
    assert "run.queue_delay_warning_ms must be a non-negative number" in message
    assert "workload.request_rate_rps must be a positive number" in message


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
