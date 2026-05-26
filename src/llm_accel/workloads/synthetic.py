from __future__ import annotations


def synthetic_prompt(input_tokens: int, seed: int = 0) -> str:
    if input_tokens <= 0:
        raise ValueError("input_tokens must be positive")
    words = [
        "benchmark",
        "latency",
        "throughput",
        "memory",
        "batching",
        "cache",
        "tokens",
        "serving",
    ]
    generated = [words[(index + seed) % len(words)] for index in range(input_tokens)]
    return " ".join(generated)


def prompt_batch(count: int, input_tokens: int, seed: int = 0) -> list[str]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [synthetic_prompt(input_tokens, seed + index) for index in range(count)]
