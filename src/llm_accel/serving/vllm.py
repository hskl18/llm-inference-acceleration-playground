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
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    speculative_model: str | None = None
    num_speculative_tokens: int | None = None

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
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        if self.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")
        if self.max_num_batched_tokens:
            args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.max_num_seqs:
            args.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.speculative_model:
            args.extend(["--speculative-model", self.speculative_model])
        if self.num_speculative_tokens:
            args.extend(["--num-speculative-tokens", str(self.num_speculative_tokens)])
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
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    speculative_model: str | None = None,
    num_speculative_tokens: int | None = None,
) -> VllmServerCommand:
    if not model:
        raise ValueError("model must be provided")
    if port <= 0:
        raise ValueError("port must be positive")
    if gpu_memory_utilization is not None and not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if max_num_seqs is not None and max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if num_speculative_tokens is not None and num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive")
    if num_speculative_tokens is not None and not speculative_model:
        raise ValueError("speculative_model is required when num_speculative_tokens is provided")
    return VllmServerCommand(
        model=model,
        host=host,
        port=port,
        dtype=dtype,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
    )


def _shell_quote(value: str) -> str:
    if all(char.isalnum() or char in "._-/:=" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
