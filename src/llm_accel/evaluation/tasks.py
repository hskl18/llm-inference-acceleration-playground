from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_accel.evaluation.io import write_mapping_jsonl
from llm_accel.evaluation.validators import normalize_validator, validate_output
from llm_accel.metrics.execution_identity import execution_identity
from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.metrics.schemas import SCHEMA_VERSION
from llm_accel.serving.openai_client import OpenAICompatibleClient


def load_task_specs(path: str | Path) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no} must contain valid JSON: {exc.msg}") from exc
        spec = _normalize_task_spec(payload, index=len(specs), context=f"line {line_no}")
        case_id = str(spec["id"])
        if case_id in seen_ids:
            raise ValueError(f"line {line_no} duplicate task id {case_id!r}")
        seen_ids.add(case_id)
        specs.append(spec)
    if not specs:
        raise ValueError("task spec file must contain at least one prompt")
    return specs


def evaluate_tasks(
    *,
    base_url: str,
    model: str,
    task_specs: list[dict[str, object]],
    output_dir: str | Path,
    backend: str = "openai-compatible",
    profile: str = "standalone",
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    max_tokens: int = 64,
    stream: bool = False,
) -> dict[str, object]:
    specs = _normalize_task_specs(task_specs)
    client = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        backend=backend,
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
    )
    checks: list[dict[str, object]] = []
    outputs: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        case_id = str(spec["id"])
        prompt = str(spec["prompt"])
        validator = spec["validator"]
        assert isinstance(validator, dict)
        validator_type = str(validator["type"])
        try:
            result = client.complete(prompt, max_tokens=max_tokens, request_index=index, stream=stream)
        except Exception as exc:
            outputs.append(
                {
                    "case_id": case_id,
                    "output_index": index,
                    "output_text": None,
                    "output_tokens": 0,
                    "ttft_ms": 0.0,
                    "total_latency_ms": 0.0,
                    "error": str(exc),
                    "token_count_method": "unavailable",
                }
            )
            checks.append(
                {
                    "case_id": case_id,
                    "output_index": index,
                    "validator_type": validator_type,
                    "score": 0.0,
                    "passed": False,
                    "reason": "request failed",
                    "error": type(exc).__name__,
                }
            )
            continue

        outputs.append(
            {
                "case_id": case_id,
                "output_index": index,
                "output_text": result.output_text,
                "output_tokens": result.output_tokens,
                "ttft_ms": result.ttft_ms,
                "total_latency_ms": result.total_latency_ms,
                "error": None,
                "token_count_method": result.token_count_method,
            }
        )
        validation = validate_output(result.output_text, validator)
        checks.append(
            {
                "case_id": case_id,
                "output_index": index,
                "validator_type": validator_type,
                "score": validation.score,
                "passed": validation.passed,
                "reason": validation.reason,
                "error": None,
            }
        )

    task_count = len(specs)
    passed_count = sum(1 for check in checks if check["passed"])
    failed_count = task_count - passed_count
    mean_score = sum(float(check["score"]) for check in checks) / task_count
    identity = execution_identity(
        profile=profile,
        model=model,
        backend=backend,
        base_url=base_url,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": identity["model"],
        "backend": identity["backend"],
        "base_url": identity["base_url"],
        "execution_identity": identity,
        "task_set_sha256": _task_set_sha256(specs),
        "task_count": task_count,
        "passed_count": passed_count,
        "failed_count": failed_count,
        "mean_score": mean_score,
        "passed": failed_count == 0,
        "checks": checks,
        "notes": [
            "Task definitions, raw model outputs, and generated summaries are stored as separate artifacts.",
            "The summary intentionally excludes prompts, expected values, JSON schemas, and generated text.",
        ],
    }
    out_dir = Path(output_dir)
    write_mapping_jsonl(out_dir / "task_specs.jsonl", specs)
    write_mapping_jsonl(out_dir / "task_outputs.jsonl", outputs)
    write_json(out_dir / "task_eval.json", report)
    _write_markdown(out_dir / "task_eval.md", report)
    write_run_manifest(
        out_dir,
        run_type="task_quality_eval",
        artifacts=["manifest.json", "task_specs.jsonl", "task_outputs.jsonl", "task_eval.json", "task_eval.md"],
    )
    return report


def _normalize_task_specs(task_specs: list[dict[str, object]]) -> list[dict[str, object]]:
    if not task_specs:
        raise ValueError("task_specs must not be empty")
    normalized: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, payload in enumerate(task_specs):
        spec = _normalize_task_spec(payload, index=index, context=f"task {index}")
        case_id = str(spec["id"])
        if case_id in seen_ids:
            raise ValueError(f"task {index} duplicate task id {case_id!r}")
        seen_ids.add(case_id)
        normalized.append(spec)
    return normalized


def _normalize_task_spec(payload: object, *, index: int, context: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must contain an object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{context} must contain a non-empty prompt string")
    case_id = payload.get("id", f"case-{index + 1:04d}")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{context} id must be a non-empty string")

    validator_payload = payload.get("validator")
    if validator_payload is not None and "expected_keywords" in payload:
        raise ValueError(f"{context} cannot contain both validator and expected_keywords")
    if validator_payload is None:
        keywords = payload.get("expected_keywords")
        if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
            raise ValueError(f"{context} expected_keywords must be a list of strings")
        validator_payload = {"type": "keywords", "expected": keywords, "case_sensitive": False}
    validator = normalize_validator(validator_payload, prompt=prompt, context=context)
    return {"id": case_id.strip(), "prompt": prompt, "validator": validator}


def _task_set_sha256(specs: list[dict[str, object]]) -> str:
    canonical = json.dumps(specs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = [
        f"| `{check['case_id']}` | `{check['validator_type']}` | {check['score']:.3f} | {check['passed']} | `{check['reason']}` | `{check['error']}` |"
        for check in report["checks"]  # type: ignore[index]
    ]
    text = "\n".join(
        [
            "# Task Quality Evaluation",
            "",
            f"- Model: `{report['model']}`",
            f"- Backend: `{report['backend']}`",
            f"- Task-set SHA-256: `{report['task_set_sha256']}`",
            f"- Mean score: `{report['mean_score']:.3f}`",
            f"- Passed: `{report['passed']}`",
            "",
            "| Case | Validator | Score | Passed | Reason | Error |",
            "| --- | --- | ---: | --- | --- | --- |",
            *rows,
            "",
            "Task definitions and verbatim model outputs are stored in separate JSONL artifacts.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
