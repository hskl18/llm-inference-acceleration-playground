from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from urllib import request


@dataclass(frozen=True)
class CompletionResult:
    output_text: str
    ttft_ms: float
    total_latency_ms: float
    output_tokens: int

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return max(self.total_latency_ms - self.ttft_ms, 0.0) / (self.output_tokens - 1)


class MockOpenAIClient:
    """Deterministic local client for smoke tests and contributor onboarding."""

    def __init__(self, model: str = "mock-model", backend: str = "mock") -> None:
        self.model = model
        self.backend = backend

    def complete(self, prompt: str, max_tokens: int, request_index: int = 0, stream: bool = True) -> CompletionResult:
        input_tokens = max(len(prompt.split()), 1)
        output_tokens = max(max_tokens, 1)
        # Deterministic synthetic timings make tests stable while preserving scaling behavior.
        ttft_ms = 18.0 + input_tokens * 0.04 + request_index * 0.1 if stream else 20.0 + input_tokens * 0.04
        tpot_ms = 2.0 + min(output_tokens, 512) * 0.002
        total_latency_ms = ttft_ms + tpot_ms * max(output_tokens - 1, 0)
        time.sleep(total_latency_ms / 1000)
        prompt_terms = [term.strip(".,:;!?").lower() for term in prompt.split() if term.strip(".,:;!?")]
        seed_terms = prompt_terms[: min(8, len(prompt_terms))] or ["mock"]
        generated = [seed_terms[index % len(seed_terms)] if index < len(seed_terms) else f"tok{index}" for index in range(output_tokens)]
        output = " ".join(generated)
        return CompletionResult(
            output_text=output,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
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
    ) -> None:
        if api_kind not in {"chat", "completion"}:
            raise ValueError("api_kind must be 'chat' or 'completion'")
        self.base_url = base_url
        self.model = model
        self.backend = backend
        self.request_timeout_seconds = request_timeout_seconds
        self.api_kind = api_kind

    def complete(self, prompt: str, max_tokens: int, request_index: int = 0, stream: bool = True) -> CompletionResult:
        if self.base_url.startswith("mock://") or self.backend == "mock":
            return MockOpenAIClient(self.model, "mock").complete(prompt, max_tokens, request_index, stream)
        if stream:
            return self._complete_streaming(prompt, max_tokens)
        return self._complete_non_streaming(prompt, max_tokens)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

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
        content = self._content_from_non_streaming_choice(body["choices"][0])
        usage = body.get("usage", {})
        output_tokens = int(usage.get("completion_tokens") or max(len(content.split()), 1))
        # Non-streaming calls cannot observe real TTFT, so keep this conservative.
        ttft_ms = total_latency_ms
        return CompletionResult(
            output_text=content,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
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
        chunks: list[str] = []
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
                content = self._content_from_streaming_choice(event.get("choices", [{}])[0])
                if content and first_token_at is None:
                    first_token_at = time.perf_counter()
                if content:
                    chunks.append(content)

        completed_at = time.perf_counter()
        output_text = "".join(chunks)
        output_tokens = max(len(output_text.split()), 1)
        ttft_ms = ((first_token_at or completed_at) - started) * 1000
        total_latency_ms = (completed_at - started) * 1000
        return CompletionResult(
            output_text=output_text,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            output_tokens=output_tokens,
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
        if self.api_kind == "completion":
            payload["prompt"] = prompt
        else:
            payload["messages"] = [{"role": "user", "content": prompt}]
        return payload

    def _content_from_non_streaming_choice(self, choice: dict[str, object]) -> str:
        if self.api_kind == "completion":
            return str(choice.get("text", ""))
        message = choice.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return ""

    def _content_from_streaming_choice(self, choice: dict[str, object]) -> str:
        if self.api_kind == "completion":
            return str(choice.get("text", ""))
        delta = choice.get("delta", {})
        if isinstance(delta, dict):
            return str(delta.get("content", ""))
        return ""
