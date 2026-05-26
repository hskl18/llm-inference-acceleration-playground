from llm_accel.speculative_decoding.analysis import acceptance_curve, baseline_comparison
from llm_accel.speculative_decoding.vanilla import run_toy_speculative


def test_acceptance_curve_contains_multiple_points() -> None:
    curve = acceptance_curve(prompts=["a", "b"], lookahead=4, acceptance_mod_values=[1, 2])

    assert len(curve) == 2
    assert curve[0]["acceptance_mod"] == 1
    assert curve[0]["estimated_speedup"] > 0


def test_baseline_comparison_reports_step_delta() -> None:
    result = run_toy_speculative(["a", "b"], lookahead=4, acceptance_mod=8).to_dict()
    comparison = baseline_comparison(result)

    assert comparison["baseline"]["steps"] == result["baseline_steps"]  # type: ignore[index]
    assert comparison["speculative"]["steps"] == result["speculative_steps"]  # type: ignore[index]
    assert comparison["saved_steps"] == result["baseline_steps"] - result["speculative_steps"]
    assert comparison["estimated_speedup"] == result["estimated_speedup"]
