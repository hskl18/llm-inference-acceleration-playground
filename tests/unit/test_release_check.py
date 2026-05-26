from scripts.release_check import changelog_has_version, check_release_metadata, read_package_version, read_pyproject_version, release_gate_commands


def test_release_metadata_versions_match_changelog() -> None:
    errors = check_release_metadata()

    assert errors == []


def test_release_metadata_readers() -> None:
    version = read_pyproject_version()

    assert read_package_version() == version
    assert changelog_has_version(version)


def test_release_gate_checks_packaging_and_console_script() -> None:
    commands = release_gate_commands("python")

    assert ["python", "-m", "pip", "install", "-e", ".", "--dry-run"] in commands
    assert ["llm-accel", "--help"] in commands
    assert ["python", "-m", "pytest"] in commands
    assert ["python", "scripts/smoke.py"] in commands
