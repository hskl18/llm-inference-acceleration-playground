from __future__ import annotations


BACKEND_CAPABILITIES: dict[str, dict[str, object]] = {
    "mock": {
        "streaming": True,
        "gpu_memory": False,
        "quantization_modes": ["none", "int8", "int4"],
        "notes": "Deterministic local backend for smoke tests; metrics are synthetic.",
    },
    "vllm": {
        "streaming": True,
        "gpu_memory": True,
        "quantization_modes": ["none", "awq", "gptq", "fp8", "int8", "int4"],
        "optimization_features": ["paged_attention", "continuous_batching", "prefix_caching", "chunked_prefill", "speculative_decoding"],
        "notes": "Capability depends on installed vLLM version, model, and hardware.",
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
        "quantization_modes": ["none", "fp8", "int8", "int4"],
        "optimization_features": ["inflight_batching", "paged_kv_cache", "kv_cache_reuse", "speculative_decoding"],
        "notes": "TensorRT-LLM capabilities depend on built engine, TensorRT-LLM version, GPU generation, and serving stack.",
    },
    "tgi": {
        "streaming": True,
        "gpu_memory": True,
        "quantization_modes": ["unknown"],
        "optimization_features": ["continuous_batching"],
        "notes": "Hugging Face TGI can expose OpenAI-compatible routes in recent deployments; feature support is deployment-specific.",
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
