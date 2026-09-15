from __future__ import annotations

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, TextIO, cast

from llm_accel.metrics.schemas import RequestMetrics


def _prepare_artifact_parent(path: Path) -> None:
    current = path.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"artifact parent path must not contain a symlink: {current}")
        if current.exists() or current == current.parent:
            break
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError(f"artifact parent path must not be a symlink: {path.parent}")


def write_json(path: Path, payload: object) -> None:
    write_json_atomic(path, payload)


@contextmanager
def _atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    _prepare_artifact_parent(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yield cast(TextIO, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_text_atomic(path: Path, payload: str) -> None:
    with _atomic_text_writer(path) as handle:
        handle.write(payload)


def write_lines_atomic(path: Path, lines: Iterable[str]) -> None:
    with _atomic_text_writer(path) as handle:
        for line in lines:
            handle.write(line)


def write_json_atomic(path: Path, payload: object) -> None:
    with _atomic_text_writer(path) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    _prepare_artifact_parent(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_jsonl(path: Path, records: Iterable[RequestMetrics]) -> None:
    with _atomic_text_writer(path) as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def write_request_csv(path: Path, records: Iterable[RequestMetrics]) -> None:
    fieldnames = [
        "request_id",
        "model",
        "backend",
        "input_tokens",
        "output_tokens",
        "concurrency",
        "ttft_ms",
        "tpot_ms",
        "total_latency_ms",
        "token_count_method",
        "completed",
        "error",
        "started_offset_ms",
        "completed_offset_ms",
        "scheduled_offset_ms",
        "dispatch_offset_ms",
        "queue_delay_ms",
        "end_to_end_latency_ms",
    ]
    with _atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
