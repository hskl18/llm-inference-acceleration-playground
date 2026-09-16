# Backend Capabilities

Backend capabilities describe what the local tooling knows about a backend. They are not a substitute for runtime feature detection.

```bash
llm-accel backend list
llm-accel backend show --backend vllm
llm-accel backend show --backend sglang
llm-accel backend profile --backend vllm --base-url http://localhost:8000/v1
```

Known quantization modes use each backend's own method names.
For vLLM they come from `QuantizationMethods` in `vllm/model_executor/layers/quantization/__init__.py` (v0.29.0), so `awq`, `gptq`, `fp8`, and `compressed-tensors` are modes while `int8` and `int4` are not.
The mock backend never quantizes, and TensorRT-LLM selects quantization when the engine is built rather than through a serving flag, so it reports `unknown`.

The capability matrix records:

- streaming support
- GPU memory visibility
- known quantization modes
- known optimization features
- backend-specific notes
- adapter status and required environment

Current named backends:

| Backend | Client path | Example optimization metadata |
| --- | --- | --- |
| `mock` | deterministic local mock | synthetic workflow validation |
| `vllm` | OpenAI-compatible HTTP | paged attention, continuous batching, prefix caching, chunked prefill, speculative decoding |
| `sglang` | OpenAI-compatible HTTP | radix cache, continuous batching, speculative decoding, structured outputs |
| `tensorrt-llm` | OpenAI-compatible HTTP | in-flight batching, paged KV cache, KV cache reuse, speculative decoding |
| `tgi` | OpenAI-compatible HTTP | continuous batching |
| `ollama` | OpenAI-compatible HTTP under `/v1` | prompt prefix cache |
| `openai-compatible` | OpenAI-compatible HTTP | unknown server-side capabilities |

`backend profile` reports `backend_version` together with `backend_version_source`.
For a vLLM endpoint it asks the server's `GET /version` first, for Ollama it asks `GET /api/version`, and it only falls back to the client's installed package, because the benchmark client is usually not the serving host.

Ollama accepts the OpenAI request body but silently ignores `ignore_eos` and `min_tokens`, so output length cannot be forced there.
Choose a `--output-tokens` budget the model will always reach and confirm afterwards that every completed request returned the same output token count.
It does return streaming usage when asked for `stream_options.include_usage`, so token counts come from the server rather than a whitespace estimate.
It exposes no NVIDIA telemetry, so GPU memory stays unavailable and the hardware claim audit correctly refuses to treat an Ollama run as GPU evidence.

`llm-accel doctor --backend vllm` also includes optional GPU memory telemetry. Missing `nvidia-smi` is reported as unavailable, not as a test failure.

`llm-accel doctor --base-url ...` checks endpoint health with `GET /models`.
The endpoint is `healthy` only when `/models` returns a 2xx response with a JSON body.
A 401 or 403 response is reported as `unauthorized`, other HTTP errors or a non-JSON body as `unhealthy`, and a connection failure or timeout as `unreachable`.
For `mock://local`, health is always available and does not require network access.

Benchmark commands use `/v1/chat/completions` by default. Pass `--api-kind completion` when targeting an OpenAI-compatible `/v1/completions` server.
