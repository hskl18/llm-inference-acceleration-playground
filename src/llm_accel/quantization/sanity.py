from __future__ import annotations

from llm_accel.serving.openai_client import OpenAICompatibleClient


DEFAULT_SANITY_PROMPTS = [
    "Explain KV cache in one sentence.",
    "Name one latency-throughput tradeoff in LLM serving.",
]


def run_quality_sanity_check(
    *,
    base_url: str,
    model: str,
    backend: str,
    quantization: str,
    prompts: list[str] | None = None,
    max_tokens: int = 64,
) -> dict[str, object]:
    client = OpenAICompatibleClient(base_url=base_url, model=model, backend=backend)
    checks = []
    selected_prompts = prompts or DEFAULT_SANITY_PROMPTS
    for index, prompt in enumerate(selected_prompts):
        try:
            result = client.complete(prompt, max_tokens=max_tokens, request_index=index, stream=False)
            text = result.output_text.strip()
            checks.append(
                {
                    "prompt_index": index,
                    "non_empty": bool(text),
                    "output_chars": len(text),
                    "output_tokens": result.output_tokens,
                    "error": None,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "prompt_index": index,
                    "non_empty": False,
                    "output_chars": 0,
                    "output_tokens": 0,
                    "error": str(exc),
                }
            )

    passed = all(check["non_empty"] and check["error"] is None for check in checks)
    return {
        "backend": backend,
        "model": model,
        "quantization": quantization,
        "prompt_count": len(selected_prompts),
        "passed": passed,
        "checks": checks,
    }
