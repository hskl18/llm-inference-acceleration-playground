import json

from llm_accel.reports.comparison import compare_run_summaries


def _summary(
    path,
    concurrency: int,
    throughput: float,
    *,
    request_count: int = 8,
    hardware_label: str = "test-host",
    profile: str = "baseline",
    git_commit: str | None = None,
    command_hash: str | None = None,
) -> None:
    payload = {
        "metadata": {
            "model": "mock-model",
            "backend": "mock",
            "dtype": "fp16",
            "quantization": "none",
            "hardware_label": hardware_label,
            "optimization_profile": profile,
            "git_commit": git_commit,
            "server_command_sha256": command_hash,
            "concurrency": concurrency,
            "input_tokens": 128,
            "output_tokens": 64,
        },
        "metrics": {
            "request_count": request_count,
            "failed_count": 0,
            "latency_ms": {"p95": 10.0},
            "throughput": {"output_tokens_per_second": throughput},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_run_summaries_writes_report(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    out = tmp_path / "out"
    _summary(a, 1, 100.0)
    _summary(b, 1, 200.0)
    _summary(c, 1, 300.0)

    report = compare_run_summaries([a, b, c], out)

    assert report["summary_count"] == 3
    assert report["runs"][1]["relative_to_first"] == 2.0
    assert report["comparable"] is True
    assert report["ranking_allowed"] is True
    assert report["warnings"] == []
    assert report["profile_aggregates"][0]["optimization_profile"] == "baseline"
    assert report["profile_aggregates"][0]["repetitions"] == 3
    assert report["profile_aggregates"][0]["output_tokens_per_second"]["mean"] == 200.0
    assert (out / "comparison.json").exists()
    assert (out / "comparison.md").exists()
    assert (out / "manifest.json").exists()


def test_compare_run_summaries_warns_for_low_sample_and_incompatibility(tmp_path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    out = tmp_path / "out"
    _summary(a, 1, 100.0, request_count=2, hardware_label="host-a")
    _summary(b, 2, 200.0, request_count=2, hardware_label="host-b")

    report = compare_run_summaries([a, b], out)

    assert report["comparable"] is False
    assert report["ranking_allowed"] is False
    assert any("hardware_label differs" in warning for warning in report["warnings"])
    assert any("concurrency differs" in warning for warning in report["warnings"])
    assert any("ranking is not justified" in warning for warning in report["warnings"])
    assert report["profile_aggregates"] == []


def test_compare_rejects_different_code_commits_and_reused_profile_command(tmp_path) -> None:
    paths = []
    shared_hash = "f" * 64
    for profile, commit in [("baseline", "a" * 40), ("prefix-cache", "b" * 40)]:
        for repetition in range(3):
            path = tmp_path / f"{profile}-{repetition}.json"
            _summary(
                path,
                8,
                100.0 + repetition,
                profile=profile,
                git_commit=commit,
                command_hash=shared_hash,
            )
            paths.append(path)

    report = compare_run_summaries(paths, tmp_path / "out")

    assert report["ranking_allowed"] is False
    assert any("git_commit differs" in warning for warning in report["warnings"])
    assert any("same server command fingerprint" in warning for warning in report["warnings"])
