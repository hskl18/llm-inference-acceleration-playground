import json
from pathlib import Path

import pytest

from llm_accel.evaluation import tasks
from llm_accel.evaluation.tasks import evaluate_tasks, load_task_specs
from llm_accel.serving.openai_client import CompletionResult


def test_load_task_specs_preserves_legacy_keyword_rubric(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"prompt": "Explain KV cache.", "expected_keywords": ["kv", "cache"]}\n', encoding="utf-8")

    specs = load_task_specs(path)

    assert specs == [
        {
            "id": "case-0001",
            "prompt": "Explain KV cache.",
            "validator": {"type": "keywords", "expected": ["kv", "cache"], "case_sensitive": False},
        }
    ]


def test_load_task_specs_preserves_empty_legacy_keyword_rubric(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"prompt": "Return anything.", "expected_keywords": []}\n', encoding="utf-8")

    specs = load_task_specs(path)

    assert specs[0]["validator"] == {"type": "keywords", "expected": [], "case_sensitive": False}


def test_load_task_specs_rejects_duplicate_ids_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id":"same","prompt":"one","validator":{"type":"exact_match","expected":"one"}}\n'
        '{"id":"same","prompt":"two","validator":{"type":"exact_match","expected":"two"}}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2 duplicate task id"):
        load_task_specs(path)


class _FakeClient:
    def __init__(self, **_: object) -> None:
        pass

    def complete(self, prompt: str, **_: object) -> CompletionResult:
        if prompt == "raise":
            raise RuntimeError("endpoint unavailable")
        output = {"exact": " 42\n", "regex": "INV-2048"}[prompt]
        return CompletionResult(output_text=output, output_tokens=1, ttft_ms=1.0, total_latency_ms=2.0)


def test_evaluate_tasks_separates_specs_raw_outputs_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "OpenAICompatibleClient", _FakeClient)
    report = evaluate_tasks(
        base_url="mock://local",
        model="mock-model",
        backend="mock",
        task_specs=[
            {"id": "exact-case", "prompt": "exact", "validator": {"type": "exact_match", "expected": "42"}},
            {"id": "regex-case", "prompt": "regex", "validator": {"type": "regex", "pattern": r"INV-[0-9]{4}"}},
            {"id": "error-case", "prompt": "raise", "validator": {"type": "exact_match", "expected": "never"}},
        ],
        output_dir=tmp_path,
    )

    assert report["passed_count"] == 2
    assert report["failed_count"] == 1
    assert report["mean_score"] == pytest.approx(2 / 3)
    specs = [json.loads(line) for line in (tmp_path / "task_specs.jsonl").read_text(encoding="utf-8").splitlines()]
    outputs = [json.loads(line) for line in (tmp_path / "task_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((tmp_path / "task_eval.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert specs[0]["prompt"] == "exact"
    assert specs[0]["validator"]["expected"] == "42"
    assert outputs[0]["output_text"] == " 42\n"
    assert outputs[2]["output_text"] is None
    assert outputs[2]["error"] == "endpoint unavailable"
    assert summary["checks"][2]["error"] == "RuntimeError"
    assert set(manifest["artifacts"]) == {
        "manifest.json",
        "task_specs.jsonl",
        "task_outputs.jsonl",
        "task_eval.json",
        "task_eval.md",
    }
    forbidden_keys = {"prompt", "expected", "pattern", "schema", "output_text"}
    assert not (_all_keys(summary) & forbidden_keys)


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()
