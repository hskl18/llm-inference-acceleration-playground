from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VllmServerCommand:
    model: str
    host: str
    port: int
    dtype: str
    quantization: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None

    def argv(self) -> list[str]:
        args = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
        ]
        if self.quantization and self.quantization != "none":
            args.extend(["--quantization", self.quantization])
        if self.max_model_len:
            args.extend(["--max-model-len", str(self.max_model_len)])
        if self.gpu_memory_utilization:
            args.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        return args

    def shell_command(self) -> str:
        return " ".join(_shell_quote(part) for part in self.argv())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["argv"] = self.argv()
        payload["shell_command"] = self.shell_command()
        return payload


def build_vllm_command(
    *,
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    quantization: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
) -> VllmServerCommand:
    if not model:
        raise ValueError("model must be provided")
    if port <= 0:
        raise ValueError("port must be positive")
    if gpu_memory_utilization is not None and not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    return VllmServerCommand(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _shell_quote(value: str) -> str:
    if all(char.isalnum() or char in "._-/:=" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
