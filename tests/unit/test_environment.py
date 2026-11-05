import subprocess
from pathlib import Path

from llm_accel.metrics.environment import collect_environment_metadata, resolve_package_commit


def test_collect_environment_metadata_includes_reproducibility_fields() -> None:
    metadata = collect_environment_metadata(hardware_label="ci-smoke")

    assert metadata["python_version"]
    assert metadata["operating_system"]
    assert metadata["hardware_label"] == "ci-smoke"
    assert "git_commit" in metadata
    assert "gpu_name" in metadata
    assert "gpu_driver_version" in metadata
    assert "cuda_version" in metadata
    assert "cuda_driver_api_version" in metadata
    assert "torch_version" in metadata


ROOT = Path(__file__).resolve().parents[2]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_benchmark_commit_ignores_the_callers_repository(tmp_path: Path, monkeypatch) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "someone else's project")
    monkeypatch.chdir(tmp_path)

    metadata = collect_environment_metadata()

    assert metadata["git_commit"] == _git(ROOT, "rev-parse", "HEAD")
    assert metadata["git_commit"] != _git(tmp_path, "rev-parse", "HEAD")


def test_package_installed_inside_another_repository_has_no_commit(tmp_path: Path) -> None:
    package = tmp_path / ".venv" / "site-packages" / "llm_accel"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "someone else's project")

    assert resolve_package_commit(package) is None


def test_package_tracked_in_its_own_checkout_reports_that_commit(tmp_path: Path) -> None:
    package = tmp_path / "src" / "llm_accel"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "the package")

    assert resolve_package_commit(package) == _git(tmp_path, "rev-parse", "HEAD")
