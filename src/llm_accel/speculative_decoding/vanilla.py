from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SpeculativeResult:
    prompts: int
    lookahead: int
    draft_calls: int
    target_calls: int
    proposed_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    baseline_steps: int
    speculative_steps: int
    estimated_speedup: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_toy_speculative(prompts: list[str], lookahead: int = 4, acceptance_mod: int = 3) -> SpeculativeResult:
    if lookahead <= 0:
        raise ValueError("lookahead must be positive")
    if not prompts:
        raise ValueError("prompts must not be empty")
    if acceptance_mod <= 0:
        raise ValueError("acceptance_mod must be positive")

    proposed = len(prompts) * lookahead
    rejected = len(prompts) * (lookahead // acceptance_mod)
    accepted = proposed - rejected
    draft_calls = len(prompts)
    target_calls = len(prompts)
    baseline_steps = proposed
    speculative_steps = draft_calls + target_calls + rejected
    speedup = baseline_steps / speculative_steps if speculative_steps else 0.0
    return SpeculativeResult(
        prompts=len(prompts),
        lookahead=lookahead,
        draft_calls=draft_calls,
        target_calls=target_calls,
        proposed_tokens=proposed,
        accepted_tokens=accepted,
        rejected_tokens=rejected,
        acceptance_rate=accepted / proposed if proposed else 0.0,
        baseline_steps=baseline_steps,
        speculative_steps=speculative_steps,
        estimated_speedup=speedup,
    )
