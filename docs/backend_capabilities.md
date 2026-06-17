# Backend Capabilities

Backend capabilities describe what the local tooling knows about a backend. They are not a substitute for runtime feature detection.

```bash
llm-accel backend list
llm-accel backend show --backend vllm
llm-accel backend show --backend sglang
llm-accel backend profile --backend vllm --base-url http://localhost:8000/v1
```

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
| `openai-compatible` | OpenAI-compatible HTTP | unknown server-side capabilities |

`llm-accel doctor --backend vllm` also includes optional GPU memory telemetry. Missing `nvidia-smi` is reported as unavailable, not as a test failure.

`llm-accel doctor --base-url ...` checks endpoint health using `/models`. For `mock://local`, health is always available and does not require network access.

Benchmark commands use `/v1/chat/completions` by default. Pass `--api-kind completion` when targeting an OpenAI-compatible `/v1/completions` server.
