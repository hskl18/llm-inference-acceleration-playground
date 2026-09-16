from __future__ import annotations

from llm_accel.serving.vllm import VLLM_QUANTIZATION_METHODS


BACKEND_CAPABILITIES: dict[str, dict[str, object]] = {
    "mock": {
        "streaming": True,
        "gpu_memory": False,
        "quantization_modes": ["none"],
        "notes": "Deterministic local backend for smoke tests; it never quantizes and its metrics are synthetic.",
    },
    "vllm": {
        "streaming": True,
        "gpu_memory": True,
        "quantization_modes": ["none", *VLLM_QUANTIZATION_METHODS],
        "optimization_features": ["paged_attention", "continuous_batching", "prefix_caching", "chunked_prefill", "speculative_decoding"],
        "notes": "Quantization names come from vLLM QuantizationMethods (v0.29.0); support also depends on the installed vLLM version, model checkpoint, and hardware.",
    },
    "sglang": {
        "streaming": True,
        "gpu_memory": True,
        "quantization_modes": ["unknown"],
        "optimization_features": ["radix_cache", "continuous_batching", "speculative_decoding", "structured_outputs"],
        "notes": "SGLang exposes OpenAI-compatible APIs; exact capabilities depend on server arguments, model, and hardware.",
    },
    "tensorrt-llm": {
        "streaming": True,
        "gpu_memory": True,
        "quantization_modes": ["unknown"],
        "optimization_features": ["inflight_batching", "paged_kv_cache", "kv_cache_reuse", "speculative_decoding"],
        "notes": "TensorRT-LLM selects quantization when the engine is built, not through a serving flag, so the served mode cannot be derived from the endpoint.",
    },
    "tgi": {
        "streaming": True,
        "gpu_memory": True,
        "quantization_modes": ["unknown"],
        "optimization_features": ["continuous_batching"],
        "notes": "Hugging Face TGI can expose OpenAI-compatible routes in recent deployments; feature support is deployment-specific.",
    },
    "ollama": {
        "streaming": True,
        "gpu_memory": False,
        # Quantization is baked into the downloaded GGUF weights, not selected by a serving flag.
        "quantization_modes": ["unknown"],
        "optimization_features": ["prompt_prefix_cache"],
        "notes": (
            "Ollama exposes OpenAI-compatible routes under /v1 and its own API under /api. It reports "
            "streaming usage but ignores ignore_eos and min_tokens, and it does not expose NVIDIA GPU "
            "telemetry; on Apple Silicon it serves from unified memory."
        ),
    },
    "openai-compatible": {
        "streaming": True,
        "gpu_memory": False,
        "quantization_modes": ["unknown"],
        "optimization_features": ["unknown"],
        "notes": "Generic HTTP endpoint; server-side implementation controls actual capabilities.",
    },
}


def list_capabilities() -> dict[str, dict[str, object]]:
    return BACKEND_CAPABILITIES


def get_capability(backend: str) -> dict[str, object]:
    return BACKEND_CAPABILITIES.get(
        backend,
        {
            "streaming": "unknown",
            "gpu_memory": "unknown",
            "quantization_modes": ["unknown"],
            "optimization_features": ["unknown"],
            "notes": "Backend is not in the local capability matrix.",
        },
    )
