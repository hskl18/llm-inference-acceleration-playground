from pathlib import Path

import pytest

from llm_accel.workloads.prompts import (
    estimate_prompt_tokens,
    fixed_prompt_batch,
    load_prompt_file,
    prompt_fingerprint,
    shared_prefix_fingerprint,
    shared_prefix_tokens,
)


def test_load_prompt_file_accepts_jsonl_and_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"prompt": "json prompt"}\nplain prompt\n"string prompt"\n', encoding="utf-8")

    assert load_prompt_file(path) == ["json prompt", "plain prompt", "string prompt"]


def test_fixed_prompt_batch_cycles_without_exposing_prompt_content() -> None:
    prompts = ["first prompt", "second prompt"]

    batch = fixed_prompt_batch(prompts, 5)

    assert batch == ["first prompt", "second prompt", "first prompt", "second prompt", "first prompt"]
    assert estimate_prompt_tokens("two tokens") == 2
    assert len(prompt_fingerprint(prompts)) == 16


def test_fixed_prompt_batch_rejects_empty_prompts() -> None:
    with pytest.raises(ValueError, match="prompts must not be empty"):
        fixed_prompt_batch([], 1)


def test_shared_prefix_metadata_does_not_expose_prompt_text() -> None:
    prompts = [
        "shared system instructions answer briefly question about batching",
        "shared system instructions answer briefly question about kv cache",
    ]

    assert shared_prefix_tokens(prompts) == 7
    fingerprint = shared_prefix_fingerprint(prompts)
    assert fingerprint is not None
    assert len(fingerprint) == 16
    assert "shared system instructions" not in fingerprint
