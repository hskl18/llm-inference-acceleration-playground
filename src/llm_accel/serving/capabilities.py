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
        "quantization_modes": ["none", "awq", "gptq", "fp8"],
        "notes": "Capability depends on installed vLLM version, model, and hardware.",
    },
    "openai-compatible": {
        "streaming": True,
        "gpu_memory": False,
        "quantization_modes": ["unknown"],
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
            "notes": "Backend is not in the local capability matrix.",
        },
    )
