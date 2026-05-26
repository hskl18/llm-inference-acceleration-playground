from __future__ import annotations

from html import escape
from pathlib import Path

from llm_accel.metrics.schemas import RequestMetrics


def write_latency_svg(path: Path, records: list[RequestMetrics]) -> None:
    completed = [record for record in records if record.completed]
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 320
    padding = 48
    if not completed:
        path.write_text(_empty_svg(width, height, "No completed requests"), encoding="utf-8")
        return

    max_latency = max(record.total_latency_ms for record in completed) or 1.0
    bar_width = max((width - padding * 2) / len(completed), 1)
    bars: list[str] = []
    for index, record in enumerate(completed):
        bar_height = (record.total_latency_ms / max_latency) * (height - padding * 2)
        x = padding + index * bar_width
        y = height - padding - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#2563eb"><title>{escape(record.request_id)}: '
            f'{record.total_latency_ms:.2f} ms</title></rect>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Latency by request">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{padding}" y="28" font-family="Arial, sans-serif" font-size="18" fill="#111827">Latency by request</text>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <text x="{padding}" y="{height - 14}" font-family="Arial, sans-serif" font-size="12" fill="#374151">requests</text>
  <text x="8" y="{padding}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_latency:.1f} ms</text>
  {''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _empty_svg(width: int, height: int, message: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="48" font-family="Arial, sans-serif" font-size="16" fill="#111827">{escape(message)}</text>
</svg>
"""


def write_sweep_svg(path: Path, aggregate: dict[str, object]) -> None:
    runs = aggregate.get("runs", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    height = 360
    padding = 56
    if not runs:
        path.write_text(_empty_svg(width, height, "No sweep runs"), encoding="utf-8")
        return

    points: list[tuple[float, float, str]] = []
    max_concurrency = 1.0
    max_throughput = 1.0
    for run in runs:  # type: ignore[assignment]
        metadata = run["metadata"]
        metrics = run["metrics"]
        concurrency = float(metadata["concurrency"])
        throughput = float(metrics["throughput"]["output_tokens_per_second"])
        max_concurrency = max(max_concurrency, concurrency)
        max_throughput = max(max_throughput, throughput)
        points.append((concurrency, throughput, str(run["run_id"])))

    circles = []
    for concurrency, throughput, run_id in points:
        x = padding + (concurrency / max_concurrency) * (width - padding * 2)
        y = height - padding - (throughput / max_throughput) * (height - padding * 2)
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#059669">'
            f"<title>{escape(run_id)}: {throughput:.2f} tok/s</title></circle>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Sweep throughput by concurrency">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{padding}" y="30" font-family="Arial, sans-serif" font-size="18" fill="#111827">Throughput by concurrency</text>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <text x="{padding}" y="{height - 16}" font-family="Arial, sans-serif" font-size="12" fill="#374151">concurrency</text>
  <text x="8" y="{padding}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_throughput:.1f} tok/s</text>
  {''.join(circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_latency_throughput_svg(path: Path, aggregate: dict[str, object]) -> None:
    runs = aggregate.get("runs", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 820
    height = 360
    padding = 56
    if not runs:
        path.write_text(_empty_svg(width, height, "No sweep runs"), encoding="utf-8")
        return

    points: list[tuple[float, float, str]] = []
    max_latency = 1.0
    max_throughput = 1.0
    for run in runs:  # type: ignore[assignment]
        metrics = run["metrics"]
        latency = float(metrics["latency_ms"]["p95"])
        throughput = float(metrics["throughput"]["output_tokens_per_second"])
        max_latency = max(max_latency, latency)
        max_throughput = max(max_throughput, throughput)
        points.append((latency, throughput, str(run["run_id"])))

    circles = []
    for latency, throughput, run_id in points:
        x = padding + (latency / max_latency) * (width - padding * 2)
        y = height - padding - (throughput / max_throughput) * (height - padding * 2)
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#dc2626">'
            f"<title>{escape(run_id)}: p95={latency:.2f} ms, {throughput:.2f} tok/s</title></circle>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Latency throughput curve">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{padding}" y="30" font-family="Arial, sans-serif" font-size="18" fill="#111827">Latency-throughput curve</text>
  <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#111827" stroke-width="1"/>
  <text x="{padding}" y="{height - 16}" font-family="Arial, sans-serif" font-size="12" fill="#374151">p95 latency ms</text>
  <text x="8" y="{padding}" font-family="Arial, sans-serif" font-size="12" fill="#374151">{max_throughput:.1f} tok/s</text>
  {''.join(circles)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")
