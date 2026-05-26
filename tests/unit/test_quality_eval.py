from llm_accel.evaluation.quality import evaluate_prompts


def test_evaluate_prompts_writes_outputs(tmp_path) -> None:
    report = evaluate_prompts(
        base_url="mock://local",
        model="mock-model",
        backend="mock",
        prompts=["hello"],
        output_dir=tmp_path,
    )

    assert report["passed"] is True
    assert (tmp_path / "quality_eval.json").exists()
    assert (tmp_path / "quality_eval.md").exists()
