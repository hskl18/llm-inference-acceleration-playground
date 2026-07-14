from __future__ import annotations

import json
from pathlib import Path

from llm_accel.metrics.io import read_jsonl
from llm_accel.metrics.schemas import RequestMetrics
from llm_accel.reports.markdown import write_summary_markdown
from llm_accel.reports.plots import write_latency_svg


def regenerate_run_report(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir)
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise ValueError(f"missing summary.json in {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    generated: list[str] = []
    warnings: list[str] = []

    write_summary_markdown(path / "summary.md", summary)
    generated.append("summary.md")

    raw_path = path / "raw_requests.jsonl"
    if raw_path.exists():
        records = [_request_metrics_from_row(row) for row in read_jsonl(raw_path)]
        write_latency_svg(path / "plots" / "latency.svg", records)
        generated.append("plots/latency.svg")
    else:
        warnings.append("raw_requests.jsonl missing; latency plot was not regenerated")

    return {
        "run_dir": str(path),
        "generated": generated,
        "warnings": warnings,
    }


def _request_metrics_from_row(row: dict[str, object]) -> RequestMetrics:
    return RequestMetrics(
        request_id=str(row["request_id"]),
        model=str(row["model"]),
        backend=str(row["backend"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        concurrency=int(row["concurrency"]),
        ttft_ms=float(row["ttft_ms"]),
        tpot_ms=float(row["tpot_ms"]),
        total_latency_ms=float(row["total_latency_ms"]),
        completed=bool(row.get("completed", True)),
        error=str(row["error"]) if row.get("error") is not None else None,
        started_offset_ms=float(row.get("started_offset_ms", 0.0)),
        completed_offset_ms=float(row.get("completed_offset_ms", 0.0)),
        scheduled_offset_ms=float(row.get("scheduled_offset_ms", row.get("started_offset_ms", 0.0))),
        dispatch_offset_ms=float(row.get("dispatch_offset_ms", row.get("started_offset_ms", 0.0))),
        queue_delay_ms=float(row.get("queue_delay_ms", 0.0)),
        end_to_end_latency_ms=float(row.get("end_to_end_latency_ms", row.get("total_latency_ms", 0.0))),
    )
