import math

import pytest

from llm_accel.speculative_decoding.analysis import acceptance_curve, baseline_comparison
from llm_accel.speculative_decoding.analytic import speculative_speedup


def _closed_form_tokens(acceptance_rate: float, lookahead: int) -> float:
    return (1 - acceptance_rate ** (lookahead + 1)) / (1 - acceptance_rate)


def test_speedup_matches_the_leviathan_closed_form() -> None:
    result = speculative_speedup(acceptance_rate=0.8, lookahead=5, draft_cost_ratio=0.2)

    expected_tokens = _closed_form_tokens(0.8, 5)
    assert math.isclose(result.expected_tokens_per_target_step, expected_tokens, rel_tol=1e-12)
    assert math.isclose(result.cost_per_target_step, 5 * 0.2 + 1, rel_tol=1e-12)
    assert math.isclose(result.estimated_speedup, expected_tokens / (5 * 0.2 + 1), rel_tol=1e-12)
    assert math.isclose(result.expected_accepted_draft_tokens, expected_tokens - 1, rel_tol=1e-12)
    assert math.isclose(
        result.expected_rejected_draft_tokens, 5 - (expected_tokens - 1), rel_tol=1e-12
    )


def test_distinct_acceptance_rates_give_distinct_results() -> None:
    low = speculative_speedup(acceptance_rate=0.3, lookahead=4, draft_cost_ratio=0.2)
    high = speculative_speedup(acceptance_rate=0.4, lookahead=4, draft_cost_ratio=0.2)

    assert low.estimated_speedup < high.estimated_speedup


def test_zero_and_full_acceptance_are_finite_limits() -> None:
    never = speculative_speedup(acceptance_rate=0.0, lookahead=4, draft_cost_ratio=0.25)
    always = speculative_speedup(acceptance_rate=1.0, lookahead=4, draft_cost_ratio=0.25)

    assert never.expected_tokens_per_target_step == 1.0
    assert math.isclose(never.estimated_speedup, 1 / (4 * 0.25 + 1), rel_tol=1e-12)
    assert always.expected_tokens_per_target_step == 5.0
    assert math.isclose(always.estimated_speedup, 5 / (4 * 0.25 + 1), rel_tol=1e-12)


def test_expensive_draft_model_predicts_a_slowdown() -> None:
    result = speculative_speedup(acceptance_rate=0.2, lookahead=8, draft_cost_ratio=0.9)

    assert result.estimated_speedup < 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"acceptance_rate": 1.5, "lookahead": 4},
        {"acceptance_rate": -0.1, "lookahead": 4},
        {"acceptance_rate": 0.5, "lookahead": 0},
        {"acceptance_rate": 0.5, "lookahead": 4, "draft_cost_ratio": -1.0},
    ],
)
def test_invalid_inputs_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        speculative_speedup(**kwargs)


def test_acceptance_curve_is_monotonic_in_acceptance_rate() -> None:
    curve = acceptance_curve(lookahead=4, draft_cost_ratio=0.2, acceptance_rates=[0.1, 0.5, 0.9])

    speedups = [row["estimated_speedup"] for row in curve]
    assert len(curve) == 3
    assert speedups == sorted(speedups)


def test_baseline_comparison_reports_cost_per_target_step() -> None:
    result = speculative_speedup(acceptance_rate=0.6, lookahead=4, draft_cost_ratio=0.2).to_dict()
    comparison = baseline_comparison(result)

    assert comparison["baseline"]["tokens_per_target_step"] == 1.0
    assert comparison["speculative"]["tokens_per_target_step"] == result["expected_tokens_per_target_step"]
    assert comparison["estimated_speedup"] == result["estimated_speedup"]
    assert math.isclose(
        comparison["relative_latency_reduction"], 1 - 1 / result["estimated_speedup"], rel_tol=1e-12
    )
