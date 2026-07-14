from __future__ import annotations

from pathlib import Path

from llm_accel.metrics.io import write_json, write_text_atomic
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.speculative_decoding.vanilla import run_toy_speculative


def baseline_comparison(result: dict[str, object]) -> dict[str, object]:
    baseline_steps = int(result["baseline_steps"])
    speculative_steps = int(result["speculative_steps"])
    saved_steps = baseline_steps - speculative_steps
    relative_step_reduction = saved_steps / baseline_steps if baseline_steps else 0.0
    return {
        "baseline": {
            "name": "target-only decoding",
            "steps": baseline_steps,
        },
        "speculative": {
            "name": "toy speculative decoding",
            "steps": speculative_steps,
            "draft_calls": result["draft_calls"],
            "target_calls": result["target_calls"],
            "accepted_tokens": result["accepted_tokens"],
            "rejected_tokens": result["rejected_tokens"],
            "acceptance_rate": result["acceptance_rate"],
        },
        "estimated_speedup": result["estimated_speedup"],
        "saved_steps": saved_steps,
        "relative_step_reduction": relative_step_reduction,
        "interpretation": _baseline_interpretation(float(result["estimated_speedup"])),
    }


def acceptance_curve(
    *,
    prompts: list[str],
    lookahead: int,
    acceptance_mod_values: list[int] | None = None,
) -> list[dict[str, object]]:
    values = acceptance_mod_values or [1, 2, 3, 4, 8]
    rows: list[dict[str, object]] = []
    for acceptance_mod in values:
        result = run_toy_speculative(prompts, lookahead=lookahead, acceptance_mod=acceptance_mod)
        payload = result.to_dict()
        payload["acceptance_mod"] = acceptance_mod
        rows.append(payload)
    return rows


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
    rows = [
        f"| {row['acceptance_mod']} | {row['acceptance_rate']:.3f} | {row['estimated_speedup']:.3f} | "
        f"{row['accepted_tokens']} | {row['rejected_tokens']} |"
        for row in curve
    ]
    text = "\n".join(
        [
            "# Speculative Decoding Toy Report",
            "",
            f"- Draft model: `{payload['draft_model']}`",
            f"- Target model: `{payload['target_model']}`",
            f"- Lookahead: `{result['lookahead']}`",
            f"- Acceptance rate: `{result['acceptance_rate']:.3f}`",
            f"- Estimated speedup: `{result['estimated_speedup']:.3f}`",
            "",
            "## Acceptance Curve",
            "",
            "| Acceptance mod | Acceptance rate | Estimated speedup | Accepted tokens | Rejected tokens |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "This is a toy accounting model. It is useful for reasoning about draft quality and verification cost, not for claiming production speedups.",
            "",
        ]
    )
    write_text_atomic(path, text)


def _baseline_interpretation(estimated_speedup: float) -> str:
    if estimated_speedup > 1.0:
        return "The toy accounting predicts fewer decoding steps than target-only decoding."
    if estimated_speedup == 1.0:
        return "The toy accounting predicts parity with target-only decoding."
    return "The toy accounting predicts overhead versus target-only decoding."


def _write_baseline_markdown(path: Path, comparison: dict[str, object]) -> None:
    baseline = comparison["baseline"]  # type: ignore[index]
    speculative = comparison["speculative"]  # type: ignore[index]
    text = "\n".join(
        [
            "# Baseline Comparison",
            "",
            "| Mode | Steps | Notes |",
            "| --- | ---: | --- |",
            f"| {baseline['name']} | {baseline['steps']} | Baseline target model steps |",
            f"| {speculative['name']} | {speculative['steps']} | Draft+target calls plus rejection cost |",
            "",
            f"- Estimated speedup: `{comparison['estimated_speedup']:.3f}`",
            f"- Saved steps: `{comparison['saved_steps']}`",
            f"- Relative step reduction: `{comparison['relative_step_reduction']:.3f}`",
            f"- Interpretation: {comparison['interpretation']}",
            "",
            "This comparison is an accounting model, not a measured serving benchmark.",
            "",
        ]
    )
    write_text_atomic(path, text)
