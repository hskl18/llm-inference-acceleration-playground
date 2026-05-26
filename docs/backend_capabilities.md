# Backend Capabilities

Backend capabilities describe what the local tooling knows about a backend. They are not a substitute for runtime feature detection.

```bash
llm-accel backend list
llm-accel backend show --backend vllm
llm-accel backend profile --backend vllm --base-url http://localhost:8000/v1
```

The capability matrix records:

- streaming support
- GPU memory visibility
- known quantization modes
- backend-specific notes
- adapter status and required environment

`llm-accel doctor --backend vllm` also includes optional GPU memory telemetry. Missing `nvidia-smi` is reported as unavailable, not as a test failure.

`llm-accel doctor --base-url ...` checks endpoint health using `/models`. For `mock://local`, health is always available and does not require network access.

Benchmark commands use `/v1/chat/completions` by default. Pass `--api-kind completion` when targeting an OpenAI-compatible `/v1/completions` server.
