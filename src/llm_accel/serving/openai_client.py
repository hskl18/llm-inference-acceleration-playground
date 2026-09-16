from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib import request

from llm_accel.metrics.token_counting import (
    TOKENIZERS_ENCODE_METHOD,
    TokenCounter,
    load_token_counter,
)


DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


def bearer_auth_header(api_key_env: str = DEFAULT_API_KEY_ENV) -> dict[str, str]:
    """Read the API key from the configured environment variable; the value is never stored."""
    api_key = os.environ.get(api_key_env)
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


@dataclass(frozen=True)
class CompletionResult:
    output_text: str
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int
    input_tokens: int | None = None
    token_count_method: str = "unknown"
    # Reasoning deltas are generated tokens: they start TTFT and count as output tokens,
    # but they are kept out of output_text so quality validators only see the answer.
    reasoning_text: str = ""
    # Gaps between consecutive content-bearing stream chunks, empty for non-streaming calls.
    inter_token_latencies_ms: tuple[float, ...] = ()

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return max(self.total_latency_ms - self.ttft_ms, 0.0) / (self.output_tokens - 1)


class MockOpenAIClient:
    """Deterministic local client for smoke tests and contributor onboarding."""

    # Real waiting keeps request offsets consistent with the synthetic latencies.
    # Unit tests replace this class attribute with a no-op; reported timings do not change.
    sleep: Callable[[float], None] = staticmethod(time.sleep)

    def __init__(
        self,
        model: str = "mock-model",
        backend: str = "mock",
        request_timeout_seconds: float = 120.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.request_timeout_seconds = request_timeout_seconds
        if sleep is not None:
            self.sleep = sleep

    def complete(self, prompt: str, max_tokens: int, request_index: int = 0, stream: bool = True) -> CompletionResult:
        input_tokens = max(len(prompt.split()), 1)
        output_tokens = max(max_tokens, 1)
        # Deterministic synthetic timings make tests stable while preserving scaling behavior.
        ttft_ms = 18.0 + input_tokens * 0.04 + request_index * 0.1 if stream else 20.0 + input_tokens * 0.04
        tpot_ms = 2.0 + min(output_tokens, 512) * 0.002
        total_latency_ms = ttft_ms + tpot_ms * max(output_tokens - 1, 0)
        if total_latency_ms > self.request_timeout_seconds * 1000:
            self.sleep(self.request_timeout_seconds)
            raise TimeoutError(f"request timed out after {self.request_timeout_seconds} seconds")
        self.sleep(total_latency_ms / 1000)
        prompt_terms = [term.strip(".,:;!?").lower() for term in prompt.split() if term.strip(".,:;!?")]
        seed_terms = prompt_terms[: min(8, len(prompt_terms))] or ["mock"]
        generated = [seed_terms[index % len(seed_terms)] if index < len(seed_terms) else f"tok{index}" for index in range(output_tokens)]
        output = " ".join(generated)
        return CompletionResult(
            output_text=output,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            token_count_method="mock_synthetic",
            inter_token_latencies_ms=tuple([tpot_ms] * max(output_tokens - 1, 0)) if stream else (),
        )


class OpenAICompatibleClient:
    """Small OpenAI-compatible non-streaming client."""

    def __init__(
        self,
        base_url: str,
        model: str,
        backend: str = "openai-compatible",
        request_timeout_seconds: float = 120.0,
        api_kind: str = "chat",
        tokenizer: str | None = None,
        tokenizer_revision: str | None = None,
        token_counter: TokenCounter | None = None,
        defer_token_count: bool = False,
        ignore_eos: bool = False,
        api_key_env: str = DEFAULT_API_KEY_ENV,
    ) -> None:
        if api_kind not in {"chat", "completion"}:
            raise ValueError("api_kind must be 'chat' or 'completion'")
        self.base_url = base_url
        self.model = model
        self.backend = backend
        self.request_timeout_seconds = request_timeout_seconds
        self.api_kind = api_kind
        self.defer_token_count = defer_token_count
        self.ignore_eos = ignore_eos
        self.api_key_env = api_key_env
        self.output_token_count_method = (
            token_counter.method if token_counter is not None else TOKENIZERS_ENCODE_METHOD
        )
        self.token_counter: TokenCounter | None
        if token_counter is not None:
            self.token_counter = token_counter
        elif backend == "vllm" and tokenizer is not None and tokenizer_revision is not None:
            self.token_counter = (
                None
                if defer_token_count
                else load_token_counter(tokenizer, tokenizer_revision)
            )
        elif backend == "vllm":
            raise ValueError("vLLM benchmarks require tokenizer and tokenizer_revision")
        else:
            self.token_counter = None

    def complete(self, prompt: str, max_tokens: int, request_index: int = 0, stream: bool = True) -> CompletionResult:
        if self.base_url.startswith("mock://") or self.backend == "mock":
            return MockOpenAIClient(
                self.model,
                "mock",
                self.request_timeout_seconds,
            ).complete(prompt, max_tokens, request_index, stream)
        if stream:
            return self._complete_streaming(prompt, max_tokens)
        return self._complete_non_streaming(prompt, max_tokens)

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", **bearer_auth_header(self.api_key_env)}

    def _complete_non_streaming(self, prompt: str, max_tokens: int) -> CompletionResult:
        started = time.perf_counter()
        endpoint = self._endpoint()
        payload = self._payload(prompt, max_tokens, stream=False)
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        with request.urlopen(req, timeout=self.request_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        total_latency_ms = (time.perf_counter() - started) * 1000
        content, reasoning = self._choice_text(body["choices"][0], stream=False)
        usage = body.get("usage", {})
        output_tokens, input_tokens, method = self._token_counts(prompt, content, usage, reasoning)
        # Non-streaming calls cannot observe real TTFT, so keep this conservative.
        ttft_ms = total_latency_ms
        return CompletionResult(
            output_text=content,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            token_count_method=method,
            reasoning_text=reasoning,
        )

    def _complete_streaming(self, prompt: str, max_tokens: int) -> CompletionResult:
        started = time.perf_counter()
        endpoint = self._endpoint()
        payload = self._payload(prompt, max_tokens, stream=True)
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        first_token_at: float | None = None
        last_token_at: float | None = None
        inter_token_latencies: list[float] = []
        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        usage: dict[str, object] = {}
        with request.urlopen(req, timeout=self.request_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_usage = event.get("usage")
                if isinstance(event_usage, dict):
                    usage = event_usage
                choices = event.get("choices", [{}])
                choice = choices[0] if isinstance(choices, list) and choices else {}
                content, reasoning = self._choice_text(choice, stream=True)
                if content or reasoning:
                    arrived_at = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = arrived_at
                    else:
                        inter_token_latencies.append((arrived_at - last_token_at) * 1000)
                    last_token_at = arrived_at
                chunks.append(content)
                reasoning_chunks.append(reasoning)

        completed_at = time.perf_counter()
        output_text = "".join(chunks)
        reasoning_text = "".join(reasoning_chunks)
        output_tokens, input_tokens, method = self._token_counts(prompt, output_text, usage, reasoning_text)
        ttft_ms = ((first_token_at or completed_at) - started) * 1000
        total_latency_ms = (completed_at - started) * 1000
        return CompletionResult(
            output_text=output_text,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
            token_count_method=method,
            reasoning_text=reasoning_text,
            inter_token_latencies_ms=tuple(inter_token_latencies),
        )

    def _endpoint(self) -> str:
        if self.api_kind == "completion":
            return f"{self.base_url.rstrip('/')}/completions"
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _payload(self, prompt: str, max_tokens: int, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": 0,
        }
        if stream:
            # Every OpenAI-compatible backend needs this to report usage on a streamed response.
            payload["stream_options"] = {"include_usage": True}
        if self.ignore_eos:
            # ignore_eos skips the tokenizer EOS token; min_tokens also blocks other stop tokens,
            # so every request generates exactly max_tokens and configs stay comparable.
            payload["ignore_eos"] = True
            payload["min_tokens"] = max_tokens
        if self.api_kind == "completion":
            payload["prompt"] = prompt
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        return payload

    def _choice_text(self, choice: object, *, stream: bool) -> tuple[str, str]:
        """Return (content, reasoning) text; null, missing, or non-string fields are empty."""
        if not isinstance(choice, dict):
            return "", ""
        if self.api_kind == "completion":
            return _text(choice.get("text")), ""
        message = choice.get("delta" if stream else "message")
        if not isinstance(message, dict):
            return "", ""
        # vLLM reasoning parsers emit reasoning_content; newer releases name it reasoning.
        reasoning = _text(message.get("reasoning_content")) or _text(message.get("reasoning"))
        return _text(message.get("content")), reasoning

    def _token_counts(
        self,
        prompt: str,
        output_text: str,
        usage: object,
        reasoning_text: str = "",
    ) -> tuple[int, int, str]:
        if self.defer_token_count and self.backend == "vllm":
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if not isinstance(prompt_tokens, int):
                raise ValueError(
                    "vLLM benchmark responses must include server prompt token usage"
                )
            return (
                completion_tokens if isinstance(completion_tokens, int) else 0,
                prompt_tokens,
                f"prompt=server_usage;output={self.output_token_count_method}",
            )
        if self.token_counter is not None:
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            output_count = count_generated_tokens(self.token_counter, output_text, reasoning_text)
            if isinstance(prompt_tokens, int):
                return (
                    output_count,
                    prompt_tokens,
                    f"prompt=server_usage;output={self.token_counter.method}",
                )
            return (
                output_count,
                self.token_counter.count(prompt),
                self.token_counter.method,
            )
        if isinstance(usage, dict):
            completion_tokens = usage.get("completion_tokens")
            prompt_tokens = usage.get("prompt_tokens")
            if isinstance(completion_tokens, int) and isinstance(prompt_tokens, int):
                return completion_tokens, prompt_tokens, "server_usage"
        generated_words = len(output_text.split()) + len(reasoning_text.split())
        return max(generated_words, 1), max(len(prompt.split()), 1), "whitespace_estimate"


def count_generated_tokens(counter: TokenCounter, output_text: str, reasoning_text: str = "") -> int:
    return counter.count(output_text) + (counter.count(reasoning_text) if reasoning_text else 0)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
