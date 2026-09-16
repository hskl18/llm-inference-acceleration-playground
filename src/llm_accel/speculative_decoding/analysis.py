from __future__ import annotations

from pathlib import Path

from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.speculative_decoding.analytic import speculative_speedup


def baseline_comparison(result: dict[str, object]) -> dict[str, object]:
    """Contrast target-only decoding with the analytical speculative configuration."""
    tokens_per_step = float(result["expected_tokens_per_target_step"])
    cost_per_step = float(result["cost_per_target_step"])
    speedup = float(result["estimated_speedup"])
    return {
        "baseline": {
            "name": "target-only decoding",
            "tokens_per_target_step": 1.0,
            "cost_per_target_step": 1.0,
        },
        "speculative": {
            "name": "speculative decoding (analytical model)",
            "tokens_per_target_step": tokens_per_step,
            "cost_per_target_step": cost_per_step,
            "acceptance_rate": result["acceptance_rate"],
            "lookahead": result["lookahead"],
            "draft_cost_ratio": result["draft_cost_ratio"],
            "expected_accepted_draft_tokens": result["expected_accepted_draft_tokens"],
            "expected_rejected_draft_tokens": result["expected_rejected_draft_tokens"],
        },
        "estimated_speedup": speedup,
        "relative_latency_reduction": 1 - 1 / speedup if speedup > 0 else 0.0,
        "interpretation": _baseline_interpretation(speedup),
    }


def acceptance_curve(
    *,
    lookahead: int,
    draft_cost_ratio: float,
    acceptance_rates: list[float] | None = None,
) -> list[dict[str, object]]:
    rates = acceptance_rates or [0.1, 0.3, 0.5, 0.7, 0.9]
    return [
        speculative_speedup(
            acceptance_rate=rate,
            lookahead=lookahead,
            draft_cost_ratio=draft_cost_ratio,
        ).to_dict()
        for rate in rates
    ]


def write_speculative_reports(output_dir: str | Path, payload: dict[str, object]) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = baseline_comparison(payload["result"])  # type: ignore[arg-type]
    payload["baseline_comparison"] = comparison
    write_json(out_dir / "speculative_summary.json", payload)
    _write_markdown(out_dir / "speculative_summary.md", payload)
    write_json(out_dir / "acceptance_curve.json", payload["acceptance_curve"])
    write_json(out_dir / "baseline_comparison.json", comparison)
    _write_baseline_markdown(out_dir / "baseline_comparison.md", comparison)
    write_run_manifest(
        out_dir,
        run_type="speculative_decoding",
        artifacts=[
            "manifest.json",
            "speculative_summary.json",
            "speculative_summary.md",
            "acceptance_curve.json",
            "baseline_comparison.json",
            "baseline_comparison.md",
        ],
    )


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    result = payload["result"]
    curve = payload["acceptance_curve"]
    measured = payload.get("measured_acceptance")
    rows = [
        f"| {row['acceptance_rate']:.3f} | {row['expected_tokens_per_target_step']:.3f} | "
        f"{row['estimated_speedup']:.3f} | {row['expected_accepted_draft_tokens']:.3f} | "
        f"{row['expected_rejected_draft_tokens']:.3f} |"
        for row in curve
    ]
    measured_lines = ["- Measured acceptance: not read from a server."]
    if isinstance(measured, dict) and measured.get("acceptance_rate") is not None:
        measured_lines = [
            f"- Measured acceptance rate from vLLM `/metrics`: `{measured['acceptance_rate']:.3f}`",
            f"- Observed draft tokens per draft: `{measured.get('observed_lookahead')}`",
        ]
    elif isinstance(measured, dict):
        measured_lines = [f"- Measured acceptance unavailable: `{measured.get('error')}`"]
    text = "\n".join(
        [
            "# Speculative Decoding Analytical Report",
            "",
            f"- Draft model: `{payload['draft_model']}`",
            f"- Target model: `{payload['target_model']}`",
            f"- Model: `{result['model']}`",
            f"- Acceptance rate: `{result['acceptance_rate']:.3f}`",
            f"- Lookahead: `{result['lookahead']}`",
            f"- Draft cost ratio: `{result['draft_cost_ratio']:.3f}`",
            f"- Expected tokens per target step: `{result['expected_tokens_per_target_step']:.3f}`",
            f"- Estimated speedup: `{result['estimated_speedup']:.3f}`",
            *measured_lines,
            "",
            "## Acceptance Curve",
            "",
            "| Acceptance rate | Tokens per target step | Estimated speedup | Accepted draft tokens | Rejected draft tokens |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "This is the closed form from Leviathan, Kalman and Matias (2023), arXiv:2211.17192.",
            "It predicts a walltime improvement from acceptance rate, lookahead, and draft cost ratio.",
            "It is not a measured serving benchmark: verification overhead, batching, and memory pressure are not modelled.",
            "",
        ]
    )
    write_text_atomic(path, text)


def _baseline_interpretation(estimated_speedup: float) -> str:
    if estimated_speedup > 1.0:
        return "The model predicts fewer target model steps per token than target-only decoding."
    if estimated_speedup == 1.0:
        return "The model predicts parity with target-only decoding."
    return "The model predicts that draft cost outweighs the accepted tokens."


def _write_baseline_markdown(path: Path, comparison: dict[str, object]) -> None:
    baseline = comparison["baseline"]  # type: ignore[index]
    speculative = comparison["speculative"]  # type: ignore[index]
    text = "\n".join(
        [
            "# Baseline Comparison",
            "",
            "| Mode | Tokens per target step | Cost per target step | Notes |",
            "| --- | ---: | ---: | --- |",
            f"| {baseline['name']} | {baseline['tokens_per_target_step']:.3f} | "
            f"{baseline['cost_per_target_step']:.3f} | One target forward pass per token |",
            f"| {speculative['name']} | {speculative['tokens_per_target_step']:.3f} | "
            f"{speculative['cost_per_target_step']:.3f} | One target verification plus "
            f"{speculative['lookahead']} draft steps |",
            "",
            f"- Estimated speedup: `{comparison['estimated_speedup']:.3f}`",
            f"- Relative latency reduction: `{comparison['relative_latency_reduction']:.3f}`",
            f"- Interpretation: {comparison['interpretation']}",
            "",
            "This comparison is an analytical model, not a measured serving benchmark.",
            "",
        ]
    )
    write_text_atomic(path, text)
