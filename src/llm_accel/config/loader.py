from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


LOCAL_ENDPOINT_PREFIXES = ("mock://", "http://localhost", "http://127.0.0.1")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _load_minimal_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ConfigError(f"invalid config line {line_no}: {raw_line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"missing key on line {line_no}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError(f"invalid indentation on line {line_no}")

        parent = stack[-1][1]
        value_text = raw_value.strip()
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value_text)
    return root


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_minimal_yaml(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    return loaded


def get_path(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def validate_benchmark_config(config: dict[str, Any]) -> None:
    errors: list[str] = []

    for section in ["run", "endpoint", "model", "workload"]:
        if not isinstance(config.get(section), dict):
            errors.append(f"{section} section must be a mapping")

    _require_string(config, "endpoint.base_url", errors)
    api_kind = get_path(config, "endpoint.api_kind", "chat")
    if api_kind not in {"chat", "completion"}:
        errors.append("endpoint.api_kind must be 'chat' or 'completion'")
    _require_string(config, "model.name", errors)
    _require_positive_int(config, "run.measured_requests", errors)
    _require_non_negative_int(config, "run.warmup_requests", errors, required=False)
    _require_positive_number(config, "run.timeout_seconds", errors, required=False)
    if get_path(config, "workload.prompts_path") is None:
        _require_positive_int_list(config, "workload.input_tokens", errors)
    else:
        _require_string(config, "workload.prompts_path", errors)
        if get_path(config, "workload.input_tokens") is not None:
            _require_positive_int_list(config, "workload.input_tokens", errors)
    _require_positive_int_list(config, "workload.output_tokens", errors)
    _require_positive_int_list(config, "workload.concurrency", errors)
    request_schedule = get_path(config, "workload.request_schedule", "closed-loop")
    if request_schedule not in {"closed-loop", "open-loop"}:
        errors.append("workload.request_schedule must be 'closed-loop' or 'open-loop'")
    request_rate_rps = get_path(config, "workload.request_rate_rps")
    if request_schedule == "open-loop" and request_rate_rps is None:
        errors.append("workload.request_rate_rps is required for open-loop scheduling")
    if request_rate_rps is not None and (
        not isinstance(request_rate_rps, (int, float))
        or isinstance(request_rate_rps, bool)
        or not math.isfinite(request_rate_rps)
        or request_rate_rps <= 0
    ):
        errors.append("workload.request_rate_rps must be a positive number")
    client_processes = get_path(config, "run.client_processes", 1)
    if not isinstance(client_processes, int) or isinstance(client_processes, bool) or client_processes <= 0:
        errors.append("run.client_processes must be a positive integer")
    else:
        concurrencies = get_path(config, "workload.concurrency", [])
        concurrency_values = concurrencies if isinstance(concurrencies, list) else [concurrencies]
        if concurrency_values and all(isinstance(value, int) for value in concurrency_values):
            if client_processes > min(concurrency_values):
                errors.append("run.client_processes must not exceed any workload.concurrency value")
    queue_delay_warning_ms = get_path(config, "run.queue_delay_warning_ms", 10.0)
    if (
        not isinstance(queue_delay_warning_ms, (int, float))
        or isinstance(queue_delay_warning_ms, bool)
        or not math.isfinite(queue_delay_warning_ms)
        or queue_delay_warning_ms < 0
    ):
        errors.append("run.queue_delay_warning_ms must be a non-negative number")
    _reject_inline_secrets(config, errors)

    if errors:
        raise ConfigError("invalid benchmark config: " + "; ".join(errors))


def sanitize_resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_value(config)
    if not isinstance(sanitized, dict):
        raise ConfigError("config root must be a mapping")
    endpoint = sanitized.get("endpoint")
    if isinstance(endpoint, dict):
        base_url = endpoint.get("base_url")
        if isinstance(base_url, str) and not base_url.startswith(LOCAL_ENDPOINT_PREFIXES):
            endpoint["base_url"] = "redacted"
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "api_key_env":
                result[key] = item
            elif lowered in {"api_key", "token", "authorization", "secret"} or any(
                lowered.endswith(f"_{suffix}") for suffix in ["api_key", "token", "authorization", "secret"]
            ):
                result[key] = "redacted"
            else:
                result[key] = _sanitize_value(item)
        return result
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _require_string(config: dict[str, Any], path: str, errors: list[str]) -> None:
    value = get_path(config, path)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")


def _require_positive_number(config: dict[str, Any], path: str, errors: list[str], *, required: bool) -> None:
    value = get_path(config, path)
    if value is None and not required:
        return
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        errors.append(f"{path} must be a positive number")


def _require_positive_int(config: dict[str, Any], path: str, errors: list[str]) -> None:
    value = get_path(config, path)
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{path} must be a positive integer")


def _require_non_negative_int(config: dict[str, Any], path: str, errors: list[str], *, required: bool) -> None:
    value = get_path(config, path)
    if value is None and not required:
        return
    if not isinstance(value, int) or value < 0:
        errors.append(f"{path} must be a non-negative integer")


def _require_positive_int_list(config: dict[str, Any], path: str, errors: list[str]) -> None:
    value = get_path(config, path)
    values = value if isinstance(value, list) else [value]
    if not values or any(not isinstance(item, int) or item <= 0 for item in values):
        errors.append(f"{path} must be a positive integer or list of positive integers")


def _reject_inline_secrets(config: dict[str, Any], errors: list[str]) -> None:
    endpoint = config.get("endpoint", {})
    if not isinstance(endpoint, dict):
        return
    blocked = ["api_key", "token", "authorization", "secret"]
    for key in endpoint:
        lowered = str(key).lower()
        if lowered == "api_key_env":
            if not isinstance(endpoint[key], str) or not endpoint[key].strip():
                errors.append("endpoint.api_key_env must name an environment variable")
            continue
        if any(secret_key == lowered or lowered.endswith(f"_{secret_key}") for secret_key in blocked):
            errors.append(f"endpoint.{key} must not contain an inline secret; use api_key_env")
