import pytest

from llm_accel.evaluation.validators import normalize_validator, validate_output


def test_exact_match_validator_uses_explicit_normalization() -> None:
    strict = normalize_validator(
        {"type": "exact_match", "expected": "Answer", "strip": False, "case_sensitive": True},
        prompt="prompt",
        context="case",
    )
    normalized = normalize_validator(
        {"type": "exact_match", "expected": "Answer", "strip": True, "case_sensitive": False},
        prompt="prompt",
        context="case",
    )

    assert validate_output(" answer\n", strict).passed is False
    assert validate_output(" answer\n", normalized).passed is True


def test_regex_validator_supports_fullmatch_search_and_flags() -> None:
    fullmatch = normalize_validator(
        {"type": "regex", "pattern": r"INV-[0-9]{4}", "mode": "fullmatch", "flags": []},
        prompt="prompt",
        context="case",
    )
    search = normalize_validator(
        {"type": "regex", "pattern": "answer", "mode": "search", "flags": ["IGNORECASE"]},
        prompt="prompt",
        context="case",
    )

    assert validate_output("INV-2048", fullmatch).passed is True
    assert validate_output("prefix INV-2048", fullmatch).passed is False
    assert validate_output("The ANSWER is here", search).passed is True


def test_regex_validator_rejects_invalid_patterns() -> None:
    with pytest.raises(ValueError, match="invalid regex pattern"):
        normalize_validator({"type": "regex", "pattern": "[", "mode": "search"}, prompt="prompt", context="case")


def test_json_schema_validator_uses_draft_2020_12() -> None:
    validator = normalize_validator(
        {
            "type": "json_schema",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "required": ["answer"],
                "properties": {"answer": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
        prompt="prompt",
        context="case",
    )

    assert validate_output('{"answer": 42}', validator).passed is True
    invalid_json = validate_output("```json\n{\"answer\": 42}\n```", validator)
    wrong_type = validate_output('{"answer": "42"}', validator)
    assert invalid_json.reason == "output is not valid JSON"
    assert wrong_type.reason == "JSON Schema validation failed"


def test_json_schema_validator_rejects_invalid_schema() -> None:
    with pytest.raises(ValueError, match="invalid Draft 2020-12 JSON Schema"):
        normalize_validator(
            {"type": "json_schema", "schema": {"type": "not-a-real-type"}},
            prompt="prompt",
            context="case",
        )


def test_long_context_validator_requires_late_needles_and_scores_all_answers() -> None:
    prompt = "context " * 600 + "The access codes are ORCHID-731 and COBALT-204."
    validator = normalize_validator(
        {
            "type": "long_context",
            "expected": ["ORCHID-731", "COBALT-204"],
            "min_prompt_chars": 4096,
            "min_expected_position": 0.75,
            "case_sensitive": False,
        },
        prompt=prompt,
        context="case",
    )

    partial = validate_output("The code is orchid-731.", validator)
    complete = validate_output("ORCHID-731 and COBALT-204", validator)
    assert partial.passed is False
    assert partial.score == 0.5
    assert complete.passed is True


def test_long_context_validator_rejects_short_or_early_needles() -> None:
    with pytest.raises(ValueError, match="below min_prompt_chars"):
        normalize_validator(
            {"type": "long_context", "expected": ["needle"], "min_prompt_chars": 100, "min_expected_position": 0.5},
            prompt="needle short",
            context="case",
        )
    with pytest.raises(ValueError, match="must appear at or after position"):
        normalize_validator(
            {"type": "long_context", "expected": ["needle"], "min_prompt_chars": 100, "min_expected_position": 0.75},
            prompt="needle " + "x" * 200,
            context="case",
        )
