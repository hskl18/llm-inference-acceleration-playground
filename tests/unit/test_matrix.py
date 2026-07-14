from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from llm_accel.benchmarks.matrix import run_matrix
from llm_accel.reports.comparison import compare_run_summaries
from llm_accel.reports.ranking_audit import (
    _effective_saturation_threshold,
    audit_performance_ranking,
)


ROOT = Path(__file__).resolve().parents[2]


def test_mock_matrix_persists_randomized_plan_and_resumes(tmp_path: Path) -> None:
    output_dir = tmp_path / "matrix"

    result = run_matrix(ROOT / "configs" / "optimization_matrix_mock.yaml", output_dir)

    assert result["planned_run_count"] == 15
    assert result["successful_run_count"] == 15
    assert result["execution_failed_run_count"] == 0
    assert result["complete"] is True
    assert result["ranking_allowed"] is False
    assert (output_dir / "matrix_plan.json").exists()
    assert (output_dir / "matrix_state.json").exists()
    assert (output_dir / "matrix_summary.json").exists()
    assert (output_dir / "comparison" / "comparison.json").exists()
    plan = json.loads((output_dir / "matrix_plan.json").read_text(encoding="utf-8"))
    first_repetition = [
        run["profile"] for run in plan["runs"] if run["repetition"] == 1
    ]
    assert sorted(first_repetition) == sorted(plan["profile_names"])
    assert first_repetition != sorted(first_repetition)

    resume_quality_path = output_dir / "quality" / "baseline" / "task_outputs.jsonl"
    resume_quality = resume_quality_path.read_text(encoding="utf-8")
    corrupted_quality = [json.loads(line) for line in resume_quality.splitlines()]
    corrupted_quality[0]["output_text"] = "tampered output"
    resume_quality_path.write_text(
        "\n".join(json.dumps(item) for item in corrupted_quality) + "\n",
        encoding="utf-8",
    )
    resume_specs_path = output_dir / "quality" / "baseline" / "task_specs.jsonl"
    resume_specs = resume_specs_path.read_text(encoding="utf-8")
    corrupted_specs = [json.loads(line) for line in resume_specs.splitlines()]
    corrupted_specs[0]["prompt"] = "coherently replaced task prompt"
    resume_specs_path.write_text(
        "\n".join(json.dumps(item) for item in corrupted_specs) + "\n",
        encoding="utf-8",
    )

    resumed = run_matrix(
        ROOT / "configs" / "optimization_matrix_mock.yaml",
        output_dir,
        resume=True,
    )

    assert resumed["skipped_on_resume_count"] == 15
    assert resumed["successful_run_count"] == 15
    assert resume_quality_path.read_text(encoding="utf-8") == resume_quality
    assert resume_specs_path.read_text(encoding="utf-8") == resume_specs

    original_audit = audit_performance_ranking(output_dir)
    original_codes = {blocker["code"] for blocker in original_audit["blockers"]}
    assert original_audit["publishable_performance_ranking"] is False
    assert "coordinated_omission_risk" in original_codes
    assert "single_run_audit_failed" in original_codes

    comparison_path = output_dir / "comparison" / "comparison.json"
    original_comparison = comparison_path.read_text(encoding="utf-8")
    comparison = json.loads(original_comparison)
    comparison["summary_count"] = 999
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    assert "comparison_evidence_mismatch" in _audit_codes(output_dir)
    comparison_path.write_text(original_comparison, encoding="utf-8")

    first_run = output_dir / str(plan["runs"][0]["run_id"])
    quality_bound_summary_path = first_run / "summary.json"
    quality_bound_summary = json.loads(quality_bound_summary_path.read_text(encoding="utf-8"))
    quality_bound_summary["metadata"]["quality_score"] = 0.5
    quality_bound_summary_path.write_text(json.dumps(quality_bound_summary), encoding="utf-8")
    compare_run_summaries(
        [output_dir / str(item["run_id"]) / "summary.json" for item in plan["runs"]],
        output_dir / "comparison",
    )
    assert "run_quality_binding_mismatch" in _audit_codes(output_dir)
    quality_bound_summary["metadata"]["quality_score"] = 1.0
    quality_bound_summary_path.write_text(json.dumps(quality_bound_summary), encoding="utf-8")
    compare_run_summaries(
        [output_dir / str(item["run_id"]) / "summary.json" for item in plan["runs"]],
        output_dir / "comparison",
    )

    raw_path = first_run / "raw_requests.jsonl"
    raw_path.write_text("{corrupt\n", encoding="utf-8")
    assert "invalid_raw_trace" in _audit_codes(output_dir)
    repaired = run_matrix(
        ROOT / "configs" / "optimization_matrix_mock.yaml",
        output_dir,
        resume=True,
    )
    assert repaired["skipped_on_resume_count"] == 14
    assert raw_path.read_text(encoding="utf-8") != "{corrupt\n"
    assert json.loads(raw_path.read_text(encoding="utf-8").splitlines()[0])["request_id"]

    copied_source = output_dir / str(plan["runs"][1]["run_id"])
    shutil.rmtree(first_run)
    shutil.copytree(copied_source, first_run)
    copied_repair = run_matrix(
        ROOT / "configs" / "optimization_matrix_mock.yaml",
        output_dir,
        resume=True,
    )
    assert copied_repair["skipped_on_resume_count"] == 14
    repaired_summary = json.loads((first_run / "summary.json").read_text(encoding="utf-8"))
    assert repaired_summary["metadata"]["optimization_profile"] == plan["runs"][0]["profile"]

    summary_path = first_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    environment_fingerprint = summary["metadata"].pop("environment_fingerprint")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert "environment_fingerprint_mismatch" in _audit_codes(output_dir)
    summary["metadata"]["environment_fingerprint"] = environment_fingerprint
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    original_gpu_name = summary["metadata"].get("gpu_name")
    summary["metadata"]["gpu_name"] = "tampered-gpu"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert "environment_fingerprint_mismatch" in _audit_codes(output_dir)
    summary["metadata"]["gpu_name"] = original_gpu_name
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    command_path = first_run / "server_command.txt"
    command = command_path.read_bytes()
    command_path.write_bytes(command + b"# changed\n")
    assert "server_command_mismatch" in _audit_codes(output_dir)
    command_path.write_bytes(command)

    quality_path = output_dir / "quality" / "baseline" / "task_eval.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["task_set_sha256"] = "0" * 64
    quality_path.write_text(json.dumps(quality), encoding="utf-8")
    assert "quality_result_mismatch" in _audit_codes(output_dir)

    output_path = output_dir / "quality" / "prefix-cache" / "task_outputs.jsonl"
    original_outputs = output_path.read_text(encoding="utf-8")
    outputs = [json.loads(line) for line in original_outputs.splitlines()]
    outputs[0]["output_text"] = "tampered output"
    output_path.write_text("\n".join(json.dumps(item) for item in outputs) + "\n", encoding="utf-8")
    assert "quality_check_mismatch" in _audit_codes(output_dir)
    output_path.write_text(original_outputs, encoding="utf-8")

    baseline_outputs_path = output_dir / "quality" / "baseline" / "task_outputs.jsonl"
    baseline_outputs = [
        json.loads(line) for line in baseline_outputs_path.read_text(encoding="utf-8").splitlines()
    ]
    baseline_outputs[0]["output_text"] = "tampered output"
    baseline_outputs_path.write_text(
        "\n".join(json.dumps(item) for item in baseline_outputs) + "\n",
        encoding="utf-8",
    )
    quality["checks"][0]["score"] = 0.0
    quality["checks"][0]["passed"] = False
    quality["checks"][0]["reason"] = "matched 0 required keywords"
    quality["passed_count"] = 1
    quality["failed_count"] = 1
    quality["mean_score"] = 0.5
    quality["passed"] = False
    quality_path.write_text(json.dumps(quality), encoding="utf-8")

    matrix_path = output_dir / "matrix_summary.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    baseline_quality = next(item for item in matrix["quality"] if item["profile"] == "baseline")
    baseline_quality["task_passed"] = True
    baseline_quality["quality_gate_passed"] = True
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    assert "quality_task_pass_mismatch" in _audit_codes(output_dir)

    state_path = output_dir / "matrix_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    original_plan_index = state["runs"][0]["plan_index"]
    state["runs"][0]["plan_index"] = 999
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert "matrix_state_mismatch" in _audit_codes(output_dir)
    state["runs"][0]["plan_index"] = original_plan_index
    state_path.write_text(json.dumps(state), encoding="utf-8")

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["runs"][0]["status"] = "execution_failed"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    assert "failed_repetition" in _audit_codes(output_dir)


def test_matrix_resume_rejects_changed_config(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "optimization_matrix_mock.yaml").read_text(encoding="utf-8")
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        (ROOT / "configs" / "task_eval_small.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source = source.replace(
        "tasks_path: task_eval_small.jsonl",
        f"tasks_path: {tasks_path}",
    )
    config_path = tmp_path / "matrix.yaml"
    config_path.write_text(source, encoding="utf-8")
    output_dir = tmp_path / "output"
    run_matrix(config_path, output_dir)
    original_tasks = tasks_path.read_text(encoding="utf-8")
    tasks_path.write_text(original_tasks + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config changed"):
        run_matrix(config_path, output_dir, resume=True)

    tasks_path.write_text(original_tasks, encoding="utf-8")
    config_path.write_text(source.replace("seed: 42", "seed: 43", 1), encoding="utf-8")

    with pytest.raises(ValueError, match="config changed"):
        run_matrix(config_path, output_dir, resume=True)


def test_ranking_saturation_ceiling_cannot_be_disabled_by_configuration() -> None:
    assert _effective_saturation_threshold(1_000_000.0, 1_000.0) == 10.0
    assert _effective_saturation_threshold(5.0, 1_000.0) == 5.0


def test_matrix_rejects_duplicate_workload_dimensions(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "optimization_matrix_mock.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(source.replace("input_tokens: [32]", "input_tokens: [32, 32]"), encoding="utf-8")

    with pytest.raises(ValueError, match="workload.input_tokens must not contain duplicate values"):
        run_matrix(config_path, tmp_path / "output")


def _audit_codes(output_dir: Path) -> set[str]:
    report = audit_performance_ranking(output_dir)
    assert report["publishable_performance_ranking"] is False
    return {str(blocker["code"]) for blocker in report["blockers"]}
