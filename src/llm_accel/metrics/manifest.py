from __future__ import annotations

from pathlib import Path

from llm_accel import __version__
from llm_accel.metrics.io import write_json
from llm_accel.metrics.schemas import SCHEMA_VERSION


def write_run_manifest(output_dir: str | Path, *, run_type: str, artifacts: list[str]) -> dict[str, object]:
    out_dir = Path(output_dir)
    manifest = {
        "project_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "run_type": run_type,
        "artifacts": artifacts,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
