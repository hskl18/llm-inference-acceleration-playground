from __future__ import annotations

from llm_accel.metrics.execution_identity import displayed_base_url
from llm_accel.serving.capabilities import get_capability
from llm_accel.serving.versions import resolve_backend_version


def backend_profile(backend: str, *, base_url: str = "mock://local") -> dict[str, object]:
    capability = get_capability(backend)
    backend_version, backend_version_source = resolve_backend_version(backend, base_url)
    return {
        "backend": backend,
        "backend_version": backend_version,
        "backend_version_source": backend_version_source,
        "base_url": displayed_base_url(base_url),
        "client": "OpenAICompatibleClient",
        "adapter_status": "implemented" if backend in {"mock", "openai-compatible", "vllm", "sglang", "tensorrt-llm", "tgi"} else "unknown",
        "capability": capability,
        "required_environment": _required_environment(backend),
        "notes": _notes(backend),
    }


def _required_environment(backend: str) -> list[str]:
    if backend == "mock":
        return []
    if backend == "vllm":
        return ["vLLM server exposing OpenAI-compatible /v1 endpoints", "optional OPENAI_API_KEY"]
    if backend == "sglang":
        return ["SGLang server exposing OpenAI-compatible /v1 endpoints", "optional OPENAI_API_KEY"]
    if backend == "tensorrt-llm":
        return ["TensorRT-LLM or Triton deployment exposing OpenAI-compatible /v1 endpoints", "optional OPENAI_API_KEY"]
    if backend == "tgi":
        return ["TGI deployment exposing OpenAI-compatible /v1 endpoints", "optional OPENAI_API_KEY"]
    return ["OpenAI-compatible /v1/chat/completions endpoint", "optional OPENAI_API_KEY"]


def _notes(backend: str) -> list[str]:
    if backend == "mock":
        return ["Deterministic local adapter for workflow validation.", "Does not represent hardware performance."]
    if backend == "vllm":
        return ["Use llm-accel vllm command to generate a startup command.", "Capabilities depend on local vLLM version and GPU."]
    if backend == "sglang":
        return ["Use backend metadata for OpenAI-compatible SGLang endpoints.", "Radix cache and speculative decoding require server-side configuration."]
    if backend == "tensorrt-llm":
        return ["Use backend metadata for OpenAI-compatible TensorRT-LLM endpoints.", "Engine build options and GPU generation determine available optimizations."]
    if backend == "tgi":
        return ["Use backend metadata for OpenAI-compatible TGI endpoints.", "Capabilities depend on deployment version and launch flags."]
    return ["Generic adapter for OpenAI-compatible endpoints."]
