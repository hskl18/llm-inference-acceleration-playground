from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from llm_accel.metrics.schemas import RequestMetrics


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[RequestMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def write_request_csv(path: Path, records: Iterable[RequestMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        "completed",
        "error",
        "started_offset_ms",
        "completed_offset_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
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
