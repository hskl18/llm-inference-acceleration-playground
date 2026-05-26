from pathlib import Path

from llm_accel.evaluation.tasks import evaluate_tasks, load_task_specs


def test_load_task_specs_reads_keyword_rubric(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"prompt": "Explain KV cache.", "expected_keywords": ["kv", "cache"]}\n', encoding="utf-8")

    specs = load_task_specs(path)

    assert specs == [{"prompt": "Explain KV cache.", "expected_keywords": ["kv", "cache"]}]


def test_evaluate_tasks_scores_keywords_with_mock_backend(tmp_path: Path) -> None:
    report = evaluate_tasks(
        base_url="mock://local",
        model="mock-model",
        backend="mock",
        task_specs=[{"prompt": "Explain KV cache.", "expected_keywords": ["kv", "cache"]}],
        output_dir=tmp_path,
    )

    assert report["passed"] is True
    assert report["mean_keyword_score"] == 1.0
    assert (tmp_path / "task_eval.json").exists()
    assert (tmp_path / "task_eval.md").exists()
