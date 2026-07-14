import json

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
    assert report["prompt_set_sha256"]
    assert (tmp_path / "quality_outputs.jsonl").exists()
    assert (tmp_path / "quality_eval.json").exists()
    assert (tmp_path / "quality_eval.md").exists()
    output = json.loads((tmp_path / "quality_outputs.jsonl").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "quality_eval.json").read_text(encoding="utf-8"))
    assert output["output_text"]
    assert "output_text" not in json.dumps(summary)
