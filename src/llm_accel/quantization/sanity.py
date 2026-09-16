from __future__ import annotations

import json
import math
from urllib import error as urllib_error
from urllib import request

from llm_accel.serving.openai_client import DEFAULT_API_KEY_ENV, OpenAICompatibleClient, bearer_auth_header


DEFAULT_SANITY_PROMPTS = [
    "Explain KV cache in one sentence.",
    "Name one latency-throughput tradeoff in LLM serving.",
]
# Fixed scoring text; perplexity is only comparable across modes when the text is identical.
DEFAULT_PERPLEXITY_TEXT = (
    "Paged attention stores key and value tensors in fixed-size blocks so that a serving engine "
    "can share and reuse cache pages across concurrent requests without contiguous allocation."
)


def measure_prompt_perplexity(
    *,
    base_url: str,
    model: str,
    text: str = DEFAULT_PERPLEXITY_TEXT,
    timeout_seconds: float = 60.0,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    """Perplexity of a fixed text from echoed prompt logprobs, when the backend supports them.

    Uses the OpenAI completions `echo` plus `logprobs` form: a backend that does not return
    `logprobs.token_logprobs` is reported as unsupported rather than guessed at.
    """
    if base_url.startswith("mock://"):
        return {"supported": False, "perplexity": None, "token_count": 0, "error": "mock endpoint has no logprobs"}
    payload = {
        "model": model,
        "prompt": text,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 0,
        "temperature": 0,
    }
    req = request.Request(
        f"{base_url.rstrip('/')}/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **bearer_auth_header(api_key_env)},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        logprobs = body["choices"][0]["logprobs"]["token_logprobs"]
    except (urllib_error.URLError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {"supported": False, "perplexity": None, "token_count": 0, "error": f"{type(exc).__name__}: {exc}"}
    values = [float(value) for value in logprobs if isinstance(value, (int, float))]
    if not values:
        return {"supported": False, "perplexity": None, "token_count": 0, "error": "no scored prompt tokens"}
    return {
        "supported": True,
        "perplexity": math.exp(-sum(values) / len(values)),
        "token_count": len(values),
        "error": None,
    }


def run_quality_sanity_check(
    *,
    base_url: str,
    model: str,
    backend: str,
    quantization: str,
    prompts: list[str] | None = None,
    max_tokens: int = 64,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    client = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        backend=backend,
        api_key_env=api_key_env,
    )
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
                    # Kept so a comparison can score this mode against the baseline mode's text.
                    "output_text": text,
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
                    "output_text": None,
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
