#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Draw the README's concurrency chart from the published sweep summaries.

Run `uv run scripts/plot_concurrency.py` to rewrite docs/media/concurrency.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "results/published/2026-09-16-ollama-apple-m3-pro/concurrency-sweep"
OUTPUT = ROOT / "docs/media/concurrency.png"
LEVELS = (1, 2, 4, 8)
THROUGHPUT = "throughput.output_tokens_per_second"

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
metrics = [json.loads((SWEEP / f"c{level}/repeats_summary.json").read_text())["metrics"] for level in LEVELS]
positions = range(len(LEVELS))
figure, (left, right) = plt.subplots(1, 2, figsize=(8, 3.2), facecolor="white")

means = [m[THROUGHPUT]["mean"] for m in metrics]
errors = [m[THROUGHPUT]["ci95_half_width"] for m in metrics]
left.bar(positions, means, yerr=errors, capsize=4, color="#4c78a8")
left.set_title("Output tokens/sec (95% CI, 3 runs)")
left.set_ylim(bottom=0)

attainment = [100 * m["goodput.attainment"]["mean"] for m in metrics]
bars = right.bar(positions, attainment, color="#e45756")
right.bar_label(bars, fmt="%.0f%%")
right.set_title("Requests meeting the latency budget")
right.set_ylim(0, 110)
right.set_yticks([0, 50, 100], ["0%", "50%", "100%"])

for axis in (left, right):
    axis.set_xticks(list(positions), [str(level) for level in LEVELS])
    axis.set_xlabel("Concurrent requests")

figure.tight_layout()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUTPUT, dpi=160, metadata={"Software": None})
