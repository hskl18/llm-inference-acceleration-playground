from __future__ import annotations

import hashlib
import json
from pathlib import Path


def load_prompt_file(path: str | Path) -> list[str]:
    prompt_path = Path(path)
    prompts: list[str] = []
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            prompts.append(stripped)
            continue
        if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            prompts.append(payload["prompt"])
        elif isinstance(payload, str):
            prompts.append(payload)
    return prompts


def estimate_prompt_tokens(prompt: str) -> int:
    return max(len(prompt.split()), 1)


def fixed_prompt_batch(prompts: list[str], count: int) -> list[str]:
    if not prompts:
        raise ValueError("prompts must not be empty")
    if count <= 0:
        raise ValueError("count must be positive")
    return [prompts[index % len(prompts)] for index in range(count)]


def prompt_fingerprint(prompts: list[str]) -> str:
    if not prompts:
        raise ValueError("prompts must not be empty")
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]
