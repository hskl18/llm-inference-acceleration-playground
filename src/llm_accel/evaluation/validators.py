from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


SUPPORTED_VALIDATOR_TYPES = {"keywords", "exact_match", "regex", "json_schema", "long_context"}
_REGEX_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
}


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    score: float
    reason: str | None = None


def normalize_validator(raw: object, *, prompt: str, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} validator must be an object")
    validator_type = raw.get("type")
    if validator_type not in SUPPORTED_VALIDATOR_TYPES:
        raise ValueError(f"{context} validator type must be one of {sorted(SUPPORTED_VALIDATOR_TYPES)}")

    if validator_type == "keywords":
        expected_raw = raw.get("expected")
        if not isinstance(expected_raw, list) or not all(isinstance(item, str) and item for item in expected_raw):
            raise ValueError(f"{context} expected must be a list of non-empty strings")
        expected = list(expected_raw)
        return {"type": "keywords", "expected": expected, "case_sensitive": _boolean(raw, "case_sensitive", False, context)}

    if validator_type == "exact_match":
        expected = raw.get("expected")
        if not isinstance(expected, str):
            raise ValueError(f"{context} exact_match expected must be a string")
        return {
            "type": "exact_match",
            "expected": expected,
            "strip": _boolean(raw, "strip", True, context),
            "case_sensitive": _boolean(raw, "case_sensitive", True, context),
        }

    if validator_type == "regex":
        pattern = raw.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{context} regex pattern must be a non-empty string")
        mode = raw.get("mode", "fullmatch")
        if mode not in {"fullmatch", "search"}:
            raise ValueError(f"{context} regex mode must be 'fullmatch' or 'search'")
        flags = raw.get("flags", [])
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise ValueError(f"{context} regex flags must be a list of strings")
        unknown_flags = sorted(set(flags) - set(_REGEX_FLAGS))
        if unknown_flags:
            raise ValueError(f"{context} unsupported regex flags: {', '.join(unknown_flags)}")
        try:
            re.compile(pattern, _regex_flags(flags))
        except re.error as exc:
            raise ValueError(f"{context} invalid regex pattern: {exc}") from exc
        return {"type": "regex", "pattern": pattern, "mode": mode, "flags": flags}

    if validator_type == "json_schema":
        schema = raw.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(f"{context} json_schema schema must be an object")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"{context} invalid Draft 2020-12 JSON Schema: {exc.message}") from exc
        return {"type": "json_schema", "schema": schema}

    expected = _string_list(raw.get("expected"), context=context, field="expected")
    min_prompt_chars = raw.get("min_prompt_chars")
    if not isinstance(min_prompt_chars, int) or isinstance(min_prompt_chars, bool) or min_prompt_chars <= 0:
        raise ValueError(f"{context} long_context min_prompt_chars must be a positive integer")
    min_expected_position = raw.get("min_expected_position")
    if not isinstance(min_expected_position, (int, float)) or isinstance(min_expected_position, bool):
        raise ValueError(f"{context} long_context min_expected_position must be a number")
    min_expected_position = float(min_expected_position)
    if not math.isfinite(min_expected_position) or not 0.0 <= min_expected_position <= 1.0:
        raise ValueError(f"{context} long_context min_expected_position must be between 0 and 1")
    case_sensitive = _boolean(raw, "case_sensitive", False, context)
    if len(prompt) < min_prompt_chars:
        raise ValueError(f"{context} long_context prompt has {len(prompt)} characters, below min_prompt_chars {min_prompt_chars}")
    prompt_for_match = prompt if case_sensitive else prompt.casefold()
    minimum_offset = int(len(prompt) * min_expected_position)
    for item in expected:
        item_for_match = item if case_sensitive else item.casefold()
        if prompt_for_match.rfind(item_for_match) < minimum_offset:
            raise ValueError(f"{context} long_context expected value must appear at or after position {min_expected_position:.3f}")
    return {
        "type": "long_context",
        "expected": expected,
        "min_prompt_chars": min_prompt_chars,
        "min_expected_position": min_expected_position,
        "case_sensitive": case_sensitive,
    }


def validate_output(output_text: str, validator: dict[str, object]) -> ValidationResult:
    validator_type = str(validator["type"])
    if validator_type == "keywords":
        return _validate_expected_strings(output_text, validator, label="keyword")
    if validator_type == "exact_match":
        expected = str(validator["expected"])
        actual = output_text
        if validator.get("strip"):
            expected = expected.strip()
            actual = actual.strip()
        if not validator.get("case_sensitive"):
            expected = expected.casefold()
            actual = actual.casefold()
        passed = actual == expected
        return ValidationResult(passed=passed, score=1.0 if passed else 0.0, reason=None if passed else "exact match failed")
    if validator_type == "regex":
        pattern = re.compile(str(validator["pattern"]), _regex_flags(list(validator.get("flags", []))))
        matched = pattern.fullmatch(output_text) if validator.get("mode") == "fullmatch" else pattern.search(output_text)
        passed = matched is not None
        return ValidationResult(passed=passed, score=1.0 if passed else 0.0, reason=None if passed else "regex did not match")
    if validator_type == "json_schema":
        try:
            instance = json.loads(output_text.strip())
        except json.JSONDecodeError:
            return ValidationResult(passed=False, score=0.0, reason="output is not valid JSON")
        errors = list(Draft202012Validator(validator["schema"]).iter_errors(instance))
        if not errors:
            return ValidationResult(passed=True, score=1.0)
        return ValidationResult(passed=False, score=0.0, reason="JSON Schema validation failed")
    if validator_type == "long_context":
        return _validate_expected_strings(output_text, validator, label="long-context answer")
    raise ValueError(f"unsupported validator type: {validator_type}")


def _validate_expected_strings(output_text: str, validator: dict[str, object], *, label: str) -> ValidationResult:
    expected = [str(item) for item in validator["expected"]]
    if not expected:
        return ValidationResult(passed=True, score=1.0)
    actual = output_text if validator.get("case_sensitive") else output_text.casefold()
    matched = 0
    for item in expected:
        item_for_match = item if validator.get("case_sensitive") else item.casefold()
        if item_for_match in actual:
            matched += 1
    score = matched / len(expected)
    passed = matched == len(expected)
    reason = None if passed else f"matched {matched} of {len(expected)} required {label}s"
    return ValidationResult(passed=passed, score=score, reason=reason)


def _string_list(value: object, *, context: str, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{context} {field} must be a non-empty list of non-empty strings")
    return list(value)


def _boolean(raw: dict[str, object], key: str, default: bool, context: str) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {key} must be a boolean")
    return value


def _regex_flags(flags: list[object]) -> int:
    value = 0
    for flag in flags:
        value |= _REGEX_FLAGS[str(flag)]
    return value
