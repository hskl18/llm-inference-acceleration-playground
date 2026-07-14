import json

import pytest

from llm_accel.metrics.optimization_profile import (
    create_optimization_profile,
    load_optimization_profile,
    optimization_profile_from_dict,
    write_optimization_profile,
)


def _profile(**overrides):
    values = {
        "name": "baseline",
        "backend": "vllm",
        "backend_version": "0.10.0",
        "server_command": "python -m vllm.entrypoints.openai.api_server --model model\n",
        "model": "model",
        "model_revision": "a" * 40,
        "tokenizer": "model",
        "tokenizer_revision": "b" * 40,
        "dtype": "float16",
        "quantization": "none",
        "environment_fingerprint": "environment-a",
        "prefix_cache": False,
        "chunked_prefill": False,
        "max_num_batched_tokens": 4096,
        "max_num_seqs": 64,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
    }
    values.update(overrides)
    return create_optimization_profile(**values)


def test_profile_round_trips_through_artifact(tmp_path) -> None:
    profile = _profile()

    path = write_optimization_profile(tmp_path, profile)
    loaded = load_optimization_profile(path)

    assert path.name == "optimization_profile.json"
    assert loaded == profile
    assert loaded.server_command_sha256 == profile.server_command_sha256
    assert len(loaded.semantic_fingerprint) == 64
    assert len(loaded.treatment_fingerprint) == 64


def test_display_name_does_not_change_semantic_fingerprint() -> None:
    baseline = _profile(name="baseline")
    alias = _profile(name="control")

    assert baseline.semantic_fingerprint == alias.semantic_fingerprint
    assert baseline.treatment_fingerprint == alias.treatment_fingerprint


def test_material_treatment_change_updates_fingerprints() -> None:
    baseline = _profile()
    prefix = _profile(
        server_command=(
            "python -m vllm.entrypoints.openai.api_server --model model "
            "--enable-prefix-caching\n"
        ),
        prefix_cache=True,
    )

    assert baseline.semantic_fingerprint != prefix.semantic_fingerprint
    assert baseline.treatment_fingerprint != prefix.treatment_fingerprint


def test_exact_command_bytes_are_hashed() -> None:
    newline = _profile(server_command="python -m server\n")
    no_newline = _profile(server_command="python -m server")

    assert newline.server_command_argv == no_newline.server_command_argv
    assert newline.server_command_sha256 != no_newline.server_command_sha256
    assert newline.treatment_fingerprint != no_newline.treatment_fingerprint


def test_tampered_profile_fingerprint_is_rejected() -> None:
    payload = _profile().to_dict()
    payload["semantic_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="semantic fingerprint"):
        optimization_profile_from_dict(payload)


def test_tampered_server_command_hash_is_rejected(tmp_path) -> None:
    path = write_optimization_profile(tmp_path, _profile())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["server_command"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="server command SHA-256"):
        load_optimization_profile(path)


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision"])
def test_model_and_tokenizer_revisions_must_be_immutable(field) -> None:
    with pytest.raises(ValueError, match="lowercase hexadecimal revision"):
        _profile(**{field: "main"})
