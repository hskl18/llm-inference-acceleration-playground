from pathlib import Path

from llm_accel import __version__
from scripts.release_check import (
    check_cli_version_output,
    check_pyproject_version_wiring,
    check_release_metadata,
    read_package_version,
    read_pyproject_version,
    release_gate_commands,
)


def test_release_metadata_has_one_authoritative_version() -> None:
    errors = check_release_metadata()

    assert errors == []
    assert check_pyproject_version_wiring() == []


def test_release_metadata_readers() -> None:
    version = read_pyproject_version()

    assert read_package_version() == version
    assert __version__ == version == "0.2.0"


def test_release_gate_checks_packaging_and_console_script() -> None:
    commands = release_gate_commands("python")

    assert ["python", "-m", "pip", "install", "-e", "."] in commands
    assert ["llm-accel", "--version"] in commands
    assert ["llm-accel", "--help"] in commands
    assert ["python", "-m", "pytest"] in commands
    assert ["python", "scripts/smoke.py"] in commands
    assert check_cli_version_output("llm-accel 0.2.0\n") is None
    assert "expected 'llm-accel 0.2.0'" in str(check_cli_version_output("llm-accel 0.1.0\n"))


def test_release_metadata_rejects_competing_static_version(tmp_path: Path) -> None:
    (tmp_path / "src/llm_accel").mkdir(parents=True)
    (tmp_path / "src/llm_accel/_version.py").write_text('__version__ = "0.2.0"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "example"
version = "9.9.9"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "llm_accel._version.__version__"}
""",
        encoding="utf-8",
    )

    assert "pyproject.toml must not contain a static [project] version" in check_release_metadata(tmp_path)
