from __future__ import annotations

import random


# Short common words keep whitespace-estimated prompt length close to model token counts.
VOCABULARY = (
    "able about above across after again against age air all also always among and answer any",
    "area arm army art ask away back bad bag ball bank base bear beat bed been before begin",
    "behind believe below best better between big bird black blood blue board boat body book",
    "born both box boy bring broad brother build burn business busy buy call came camp can",
    "care carry case catch cause cell center century certain chair chance change check child",
    "choose city class clean clear close cold color come common company cool copy corn cost",
    "could count country course cover cross crowd cut dark day dead deal dear decide deep",
    "describe desert design develop die differ direct discuss divide dog door down draw dream",
    "dress drink drive drop dry during each early earth east easy eat edge effect egg eight",
    "either electric else end enemy energy enough enter equal even evening event ever every",
)
_WORDS = tuple(word for line in VOCABULARY for word in line.split())


def synthetic_prompt(input_tokens: int, seed: int = 0, index: int = 0) -> str:
    """Build one deterministic prompt from a seeded PRNG over a fixed vocabulary.

    Prompts drawn for different indices share no prefix beyond chance, so servers with
    automatic prefix caching (vLLM enables it by default) cannot serve them from cache.
    """
    if input_tokens <= 0:
        raise ValueError("input_tokens must be positive")
    rng = random.Random(f"llm-accel-synthetic:{seed}:{index}")
    return " ".join(rng.choice(_WORDS) for _ in range(input_tokens))


def prompt_batch(count: int, input_tokens: int, seed: int = 0, first_index: int = 0) -> list[str]:
    """Build `count` prompts starting at `first_index`.

    Measured requests use indices 0 to count-1, warmup requests use later indices, so warmup
    never pre-populates a cache entry for a measured prompt and changing the warmup count
    does not change the measured workload.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if first_index < 0:
        raise ValueError("first_index must be non-negative")
    return [synthetic_prompt(input_tokens, seed, first_index + offset) for offset in range(count)]
