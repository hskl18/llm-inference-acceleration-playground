"""Analytical speculative decoding model from Leviathan, Kalman and Matias (2023).

"Fast Inference from Transformers via Speculative Decoding" (arXiv:2211.17192) gives, for
acceptance rate `a`, lookahead `g` draft tokens per target step, and draft-to-target cost ratio `c`:

    expected tokens per target step = (1 - a^(g+1)) / (1 - a)
    expected walltime improvement   = (1 - a^(g+1)) / ((1 - a) * (g * c + 1))

This module computes exactly those quantities. It is an analytical model, not a measurement:
a real speedup depends on batch composition, kernel efficiency, and verification overhead, so
`read_vllm_acceptance` is provided to read the served acceptance rate from a vLLM endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from llm_accel.serving.openai_client import DEFAULT_API_KEY_ENV
from llm_accel.serving.vllm_server import fetch_serving_state


MODEL_NAME = "leviathan-2023-closed-form"


@dataclass(frozen=True)
class SpeculativeSpeedupModel:
    model: str
    acceptance_rate: float
    lookahead: int
    draft_cost_ratio: float
    expected_tokens_per_target_step: float
    expected_accepted_draft_tokens: float
    expected_rejected_draft_tokens: float
    cost_per_target_step: float
    estimated_speedup: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def speculative_speedup(
    *,
    acceptance_rate: float,
    lookahead: int = 4,
    draft_cost_ratio: float = 0.2,
) -> SpeculativeSpeedupModel:
    """Expected tokens per target step and walltime improvement for one speculative configuration."""
    if not isinstance(lookahead, int) or isinstance(lookahead, bool) or lookahead < 1:
        raise ValueError("lookahead must be a positive integer number of draft tokens per target step")
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError("acceptance_rate must be between 0 and 1")
    if draft_cost_ratio < 0:
        raise ValueError("draft_cost_ratio must be non-negative")

    # The geometric sum 1 + a + ... + a^g; the closed form is 0/0 at a == 1, where the limit is g+1.
    expected_tokens = (
        float(lookahead + 1)
        if acceptance_rate == 1.0
        else (1 - acceptance_rate ** (lookahead + 1)) / (1 - acceptance_rate)
    )
    cost_per_step = lookahead * draft_cost_ratio + 1
    accepted = expected_tokens - 1
    return SpeculativeSpeedupModel(
        model=MODEL_NAME,
        acceptance_rate=acceptance_rate,
        lookahead=lookahead,
        draft_cost_ratio=draft_cost_ratio,
        expected_tokens_per_target_step=expected_tokens,
        expected_accepted_draft_tokens=accepted,
        expected_rejected_draft_tokens=lookahead - accepted,
        cost_per_target_step=cost_per_step,
        estimated_speedup=expected_tokens / cost_per_step,
    )


def read_vllm_acceptance(
    base_url: str,
    *,
    timeout_seconds: float = 5.0,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    """Measured acceptance counters from a vLLM endpoint's Prometheus /metrics endpoint.

    Returns the counters plus the acceptance rate the analytical model would take as its input,
    so an analytical estimate can be stated against a served configuration instead of a guess.
    """
    state = fetch_serving_state(base_url, timeout_seconds=timeout_seconds, api_key_env=api_key_env)
    acceptance = state["acceptance"] if state["available"] else None
    return {
        "metrics_available": bool(state["available"]),
        "error": state["error"],
        "counters": state["counters"],
        "acceptance": acceptance,
        "acceptance_rate": acceptance["acceptance_rate"] if isinstance(acceptance, dict) else None,
        "observed_lookahead": acceptance.get("draft_tokens_per_draft") if isinstance(acceptance, dict) else None,
    }
