from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = Path("src/llm_accel/_version.py")
VERSION_ATTRIBUTE = "llm_accel._version.__version__"


def read_pyproject_version(root: Path = ROOT) -> str:
    errors = check_pyproject_version_wiring(root)
    if errors:
        raise ValueError("; ".join(errors))
    return read_package_version(root)


def check_pyproject_version_wiring(root: Path = ROOT) -> list[str]:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    errors: list[str] = []
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
    project_block = project_match.group(1) if project_match else ""
    if re.search(r'(?m)^version\s*=\s*"', project_block):
        errors.append("pyproject.toml must not contain a static [project] version")
    if not re.search(r'(?s)^dynamic\s*=\s*\[[^\]]*"version"[^\]]*\]', project_block, re.MULTILINE):
        errors.append('pyproject.toml [project] dynamic must include "version"')
    dynamic_match = re.search(r"(?ms)^\[tool\.setuptools\.dynamic\]\s*(.*?)(?=^\[|\Z)", text)
    dynamic_block = dynamic_match.group(1) if dynamic_match else ""
    expected = rf'^version\s*=\s*\{{\s*attr\s*=\s*"{re.escape(VERSION_ATTRIBUTE)}"\s*\}}\s*$'
    if not re.search(expected, dynamic_block, re.MULTILINE):
        errors.append(f"pyproject.toml dynamic version must use {VERSION_ATTRIBUTE}")
    return errors


def read_package_version(root: Path = ROOT) -> str:
    text = (root / VERSION_FILE).read_text(encoding="utf-8")
    match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError(f"{VERSION_FILE} is missing __version__")
    return match.group(1)


def check_release_metadata(root: Path = ROOT) -> list[str]:
    errors = check_pyproject_version_wiring(root)
    package_version = read_package_version(root)
    semver = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?"
    if not re.fullmatch(semver, package_version):
        errors.append(f"authoritative package version {package_version!r} is not valid SemVer")
    return errors


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def check_cli_version_output(output: str, root: Path = ROOT) -> str | None:
    expected = f"llm-accel {read_package_version(root)}"
    actual = output.strip()
    if actual != expected:
        return f"llm-accel --version printed {actual!r}; expected {expected!r}"
    return None


def run_cli_version_check(command: list[str]) -> None:
    print("+ " + " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    error = check_cli_version_output(completed.stdout)
    if error:
        print(error, file=sys.stderr)
        raise SystemExit(1)


def release_gate_commands(python_executable: str = sys.executable) -> list[list[str]]:
    return [
        [python_executable, "-m", "pip", "install", "-e", "."],
        ["llm-accel", "--version"],
        ["llm-accel", "--help"],
        [python_executable, "-m", "ruff", "check", "."],
        [python_executable, "-m", "pytest"],
        [python_executable, "scripts/smoke.py"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local release readiness checks")
    parser.add_argument("--metadata-only", action="store_true", help="Only check authoritative version metadata")
    args = parser.parse_args(argv)

    errors = check_release_metadata(ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("release metadata ok")

    if args.metadata_only:
        return 0

    for command in release_gate_commands(sys.executable):
        if command == ["llm-accel", "--version"]:
            run_cli_version_check(command)
        else:
            run(command)
    print("release check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
