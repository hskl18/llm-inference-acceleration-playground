from llm_accel.reports.validation import validate_run_dir


def test_validate_run_dir_reports_missing_manifest(tmp_path) -> None:
    result = validate_run_dir(tmp_path)

    assert result["valid"] is False
    assert "missing manifest.json" in result["errors"]


def test_validate_run_dir_requires_summary_warnings(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"schema_version":"0.1","run_type":"latency_benchmark","artifacts":["manifest.json","summary.json"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        """
{
  "schema_version": "0.1",
  "metadata": {
    "api_kind": "chat",
    "backend_version": "test",
    "project_version": "0.1.0",
    "python_version": "3.13",
    "operating_system": "test",
    "hardware_label": "test"
  },
  "metrics": {
    "request_count": 1,
    "completed_count": 1,
    "failed_count": 0,
    "timeout_count": 0,
    "latency_ms": {},
    "throughput": {}
  },
  "memory": {}
}
""",
        encoding="utf-8",
    )

    result = validate_run_dir(tmp_path)

    assert result["valid"] is False
    assert "summary.json missing warnings" in result["errors"]


def test_validate_run_dir_checks_throughput_summary_schema(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"schema_version":"0.1","run_type":"throughput_benchmark","artifacts":["manifest.json","throughput_summary.json"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "throughput_summary.json").write_text(
        """
{
  "schema_version": "0.1",
  "metadata": {},
  "throughput": {
    "output_tokens_per_second": 1.0,
    "requests_per_second": 1.0
  },
  "completed_count": 1,
  "failed_count": 0,
  "timeout_count": 0,
  "warnings": []
}
""",
        encoding="utf-8",
    )

    result = validate_run_dir(tmp_path)

    assert result["valid"] is False
    assert "throughput_summary.json.throughput missing measured_elapsed_seconds" in result["errors"]


def test_validate_run_dir_checks_quality_eval_counts(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"schema_version":"0.1","run_type":"quality_sanity_eval","artifacts":["manifest.json","quality_eval.json"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "quality_eval.json").write_text(
        """
{
  "model": "mock-model",
  "backend": "mock",
  "prompt_count": 2,
  "passed": true,
  "checks": [{}],
  "notes": []
}
""",
        encoding="utf-8",
    )

    result = validate_run_dir(tmp_path)

    assert result["valid"] is False
    assert "quality_eval.json prompt_count does not match checks length" in result["errors"]


def test_validate_run_dir_checks_quantization_comparison_counts(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"schema_version":"0.1","run_type":"quantization_comparison","artifacts":["manifest.json","quantization_comparison.json"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "quantization_comparison.json").write_text(
        """
{
  "model": "mock-model",
  "backend": "mock",
  "modes": ["none", "int8"],
  "supported_modes": ["none", "int8"],
  "runs": [{}],
  "warnings": [],
  "notes": []
}
""",
        encoding="utf-8",
    )

    result = validate_run_dir(tmp_path)

    assert result["valid"] is False
    assert "quantization_comparison.json runs length does not match modes length" in result["errors"]
