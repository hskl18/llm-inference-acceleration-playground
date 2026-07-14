from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from llm_accel.metrics.io import write_json


OPTIMIZATION_PROFILE_SCHEMA_VERSION = "0.2"
OPTIMIZATION_PROFILE_ARTIFACT = "optimization_profile.json"
_IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40,64}")


class OptimizationProfileMismatchError(ValueError):
    pass


@dataclass(frozen=True)
class OptimizationProfile:
    name: str
    backend: str
    backend_version: str
    server_command_text: str
    server_command_argv: tuple[str, ...]
    server_command_sha256: str
    model: str
    model_revision: str
    tokenizer: str
    tokenizer_revision: str
    dtype: str
    quantization: str
    environment_fingerprint: str
    prefix_cache: bool = False
    chunked_prefill: bool = False
    speculative_model: str | None = None
    speculative_model_revision: str | None = None
    num_speculative_tokens: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    schema_version: str = OPTIMIZATION_PROFILE_SCHEMA_VERSION

    @property
    def semantic_fingerprint(self) -> str:
        return _fingerprint(self._semantic_payload())

    @property
    def treatment_fingerprint(self) -> str:
        return _fingerprint(self._treatment_payload())

    def to_dict(self) -> dict[str, object]:
        payload = self.to_dict_without_fingerprints()
        payload["semantic_fingerprint"] = self.semantic_fingerprint
        payload["treatment_fingerprint"] = self.treatment_fingerprint
        return payload

    def to_dict_without_fingerprints(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "backend": {
                "name": self.backend,
                "version": self.backend_version,
            },
            "server_command": {
                "text": self.server_command_text,
                "argv": list(self.server_command_argv),
                "sha256": self.server_command_sha256,
            },
            "model": {
                "name": self.model,
                "revision": self.model_revision,
            },
            "tokenizer": {
                "name": self.tokenizer,
                "revision": self.tokenizer_revision,
            },
            "dtype": self.dtype,
            "quantization": self.quantization,
            "features": {
                "prefix_cache": self.prefix_cache,
                "chunked_prefill": self.chunked_prefill,
                "speculative": {
                    "model": self.speculative_model,
                    "model_revision": self.speculative_model_revision,
                    "num_speculative_tokens": self.num_speculative_tokens,
                },
            },
            "batching": {
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_seqs": self.max_num_seqs,
            },
            "limits": {
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            },
            "environment_fingerprint": self.environment_fingerprint,
        }

    def _semantic_payload(self) -> dict[str, object]:
        payload = self.to_dict_without_fingerprints()
        payload.pop("name")
        return payload

    def _treatment_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "server_command_sha256": self.server_command_sha256,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "features": {
                "prefix_cache": self.prefix_cache,
                "chunked_prefill": self.chunked_prefill,
                "speculative_model": self.speculative_model,
                "speculative_model_revision": self.speculative_model_revision,
                "num_speculative_tokens": self.num_speculative_tokens,
            },
            "batching": {
                "max_num_batched_tokens": self.max_num_batched_tokens,
                "max_num_seqs": self.max_num_seqs,
            },
            "limits": {
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
            },
        }

def create_optimization_profile(
    *,
    name: str,
    backend: str,
    backend_version: str,
    server_command: str | bytes,
    model: str,
    model_revision: str,
    tokenizer: str,
    tokenizer_revision: str,
    dtype: str,
    quantization: str,
    environment_fingerprint: str,
    prefix_cache: bool = False,
    chunked_prefill: bool = False,
    speculative_model: str | None = None,
    speculative_model_revision: str | None = None,
    num_speculative_tokens: int | None = None,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> OptimizationProfile:
    command_bytes = server_command.encode("utf-8") if isinstance(server_command, str) else server_command
    try:
        command_text = command_bytes.decode("utf-8")
        command_argv = tuple(shlex.split(command_text.strip()))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"server command must be valid shell-like UTF-8 text: {exc}") from exc
    if not command_argv:
        raise ValueError("server command must not be empty")
    _require_text(name, "name")
    _require_text(backend, "backend")
    _require_text(backend_version, "backend_version")
    _require_text(model, "model")
    _require_revision(model_revision, "model_revision")
    _require_text(tokenizer, "tokenizer")
    _require_revision(tokenizer_revision, "tokenizer_revision")
    _require_text(dtype, "dtype")
    _require_text(quantization, "quantization")
    _require_text(environment_fingerprint, "environment_fingerprint")
    if speculative_model is None and speculative_model_revision is not None:
        raise ValueError("speculative_model is required with speculative_model_revision")
    if speculative_model is not None:
        _require_text(speculative_model, "speculative_model")
        if speculative_model_revision is None:
            raise ValueError("speculative_model_revision is required with speculative_model")
        _require_revision(speculative_model_revision, "speculative_model_revision")
    if num_speculative_tokens is not None and speculative_model is None:
        raise ValueError("speculative_model is required with num_speculative_tokens")
    _require_optional_positive_int(num_speculative_tokens, "num_speculative_tokens")
    _require_optional_positive_int(max_num_batched_tokens, "max_num_batched_tokens")
    _require_optional_positive_int(max_num_seqs, "max_num_seqs")
    _require_optional_positive_int(max_model_len, "max_model_len")
    if gpu_memory_utilization is not None and not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    return OptimizationProfile(
        name=name,
        backend=backend,
        backend_version=backend_version,
        server_command_text=command_text,
        server_command_argv=command_argv,
        server_command_sha256=hashlib.sha256(command_bytes).hexdigest(),
        model=model,
        model_revision=model_revision,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        dtype=dtype,
        quantization=quantization,
        environment_fingerprint=environment_fingerprint,
        prefix_cache=prefix_cache,
        chunked_prefill=chunked_prefill,
        speculative_model=speculative_model,
        speculative_model_revision=speculative_model_revision,
        num_speculative_tokens=num_speculative_tokens,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def optimization_profile_from_dict(payload: Mapping[str, object]) -> OptimizationProfile:
    if payload.get("schema_version") != OPTIMIZATION_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            "optimization profile schema_version must be "
            f"{OPTIMIZATION_PROFILE_SCHEMA_VERSION!r}"
        )
    backend = _mapping(payload, "backend")
    command = _mapping(payload, "server_command")
    model = _mapping(payload, "model")
    tokenizer = _mapping(payload, "tokenizer")
    features = _mapping(payload, "features")
    speculative = _mapping(features, "speculative")
    batching = _mapping(payload, "batching")
    limits = _mapping(payload, "limits")
    command_text = _string(command, "text")
    profile = create_optimization_profile(
        name=_string(payload, "name"),
        backend=_string(backend, "name"),
        backend_version=_string(backend, "version"),
        server_command=command_text,
        model=_string(model, "name"),
        model_revision=_string(model, "revision"),
        tokenizer=_string(tokenizer, "name"),
        tokenizer_revision=_string(tokenizer, "revision"),
        dtype=_string(payload, "dtype"),
        quantization=_string(payload, "quantization"),
        environment_fingerprint=_string(payload, "environment_fingerprint"),
        prefix_cache=_boolean(features, "prefix_cache"),
        chunked_prefill=_boolean(features, "chunked_prefill"),
        speculative_model=_optional_string(speculative, "model"),
        speculative_model_revision=_optional_string(speculative, "model_revision"),
        num_speculative_tokens=_optional_int(speculative, "num_speculative_tokens"),
        max_num_batched_tokens=_optional_int(batching, "max_num_batched_tokens"),
        max_num_seqs=_optional_int(batching, "max_num_seqs"),
        max_model_len=_optional_int(limits, "max_model_len"),
        gpu_memory_utilization=_optional_number(limits, "gpu_memory_utilization"),
    )
    argv = command.get("argv")
    if not isinstance(argv, list) or argv != list(profile.server_command_argv):
        raise ValueError("optimization profile server_command.argv does not match text")
    if command.get("sha256") != profile.server_command_sha256:
        raise ValueError("optimization profile server command SHA-256 does not match text")
    if payload.get("semantic_fingerprint") != profile.semantic_fingerprint:
        raise ValueError("optimization profile semantic fingerprint does not match contents")
    if payload.get("treatment_fingerprint") != profile.treatment_fingerprint:
        raise ValueError("optimization profile treatment fingerprint does not match contents")
    return profile


def write_optimization_profile(output_dir: str | Path, profile: OptimizationProfile) -> Path:
    path = Path(output_dir) / OPTIMIZATION_PROFILE_ARTIFACT
    write_json(path, profile.to_dict())
    return path


def load_optimization_profile(path: str | Path) -> OptimizationProfile:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("optimization profile artifact must contain an object")
    return optimization_profile_from_dict(payload)


def load_bound_optimization_profile(
    run_dir: str | Path,
    inline: object,
    *,
    require_artifact: bool = False,
) -> OptimizationProfile | None:
    artifact_path = Path(run_dir) / OPTIMIZATION_PROFILE_ARTIFACT
    if require_artifact and not artifact_path.is_file():
        raise FileNotFoundError("optimization_profile.json is required")
    artifact_profile = load_optimization_profile(artifact_path) if artifact_path.exists() else None
    inline_profile = optimization_profile_from_dict(inline) if isinstance(inline, dict) else None
    if artifact_profile is not None and inline_profile is not None:
        if artifact_profile.to_dict() != inline_profile.to_dict():
            raise OptimizationProfileMismatchError(
                "summary optimization_profile_spec does not match optimization_profile.json"
            )
        return artifact_profile
    return artifact_profile or inline_profile


def fingerprint_payload(payload: Mapping[str, object]) -> str:
    return _fingerprint(payload)


def _fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_revision(value: str, field: str) -> None:
    if not isinstance(value, str) or not _IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(f"{field} must be a 40 to 64 character lowercase hexadecimal revision")


def _require_optional_positive_int(value: int | None, field: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{field} must be a positive integer")


def _mapping(payload: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"optimization profile {field} must be an object")
    return value


def _string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"optimization profile {field} must be a string")
    return value


def _optional_string(payload: Mapping[str, object], field: str) -> str | None:
    value = payload.get(field)
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"optimization profile {field} must be a string or null")


def _boolean(payload: Mapping[str, object], field: str) -> bool:
    value = payload.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"optimization profile {field} must be a boolean")
    return value


def _optional_int(payload: Mapping[str, object], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"optimization profile {field} must be an integer or null")
    return value


def _optional_number(payload: Mapping[str, object], field: str) -> float | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"optimization profile {field} must be numeric or null")
    return float(value)
