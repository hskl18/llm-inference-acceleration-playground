from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_pyproject_version(root: Path = ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError("pyproject.toml is missing [project] version")
    return match.group(1)


def read_package_version(root: Path = ROOT) -> str:
    text = (root / "src" / "llm_accel" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError("src/llm_accel/__init__.py is missing __version__")
    return match.group(1)


def changelog_has_version(version: str, root: Path = ROOT) -> bool:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.search(rf"(?m)^##\s+{re.escape(version)}(?:\s+-\s+.+)?$", text) is not None


def check_release_metadata(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    pyproject_version = read_pyproject_version(root)
    package_version = read_package_version(root)
    if pyproject_version != package_version:
        errors.append(f"pyproject.toml version {pyproject_version} does not match package __version__ {package_version}")
    if not changelog_has_version(pyproject_version, root):
        errors.append(f"CHANGELOG.md is missing a section for version {pyproject_version}")
    return errors


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def release_gate_commands(python_executable: str = sys.executable) -> list[list[str]]:
    return [
        [python_executable, "-m", "pip", "install", "-e", ".", "--dry-run"],
        ["llm-accel", "--help"],
        [python_executable, "-m", "ruff", "check", "."],
        [python_executable, "-m", "pytest"],
        [python_executable, "scripts/smoke.py"],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local release readiness checks")
    parser.add_argument("--metadata-only", action="store_true", help="Only check version and changelog metadata")
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
        run(command)
    print("release check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
