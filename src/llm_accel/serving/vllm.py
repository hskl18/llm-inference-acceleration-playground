from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass


VLLM_DTYPES = {"auto", "bfloat16", "float", "float16", "float32", "half"}
DTYPE_ALIASES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

# vllm/model_executor/layers/quantization/__init__.py::QuantizationMethods (vLLM v0.29.0).
# Plain "int8" and "int4" are not vLLM method names; vLLM accepts any string for
# --quantization and only fails when the engine resolves it, so this list is the gate.
VLLM_QUANTIZATION_METHODS = (
    "awq",
    "auto_awq",
    "fp8",
    "fbgemm_fp8",
    "fp_quant",
    "modelopt",
    "modelopt_fp4",
    "modelopt_mxfp8",
    "modelopt_mixed",
    "auto_gptq",
    "gptq",
    "gptq_marlin",
    "awq_marlin",
    "humming",
    "compressed-tensors",
    "experts_int8",
    "quark",
    "moe_wna16",
    "torchao",
    "inc",
    "mxfp4",
    "gpt_oss_mxfp4",
    "deepseek_v4_fp8",
    "online",
    "fp8_per_tensor",
    "fp8_per_block",
    "fp8_per_channel",
    "int8_per_channel_weight_only",
    "nvfp4_per_token",
    "mxfp8",
)

# Speculative methods that propose tokens from a separate draft checkpoint.
VLLM_SPECULATIVE_METHODS_REQUIRING_MODEL = frozenset({"draft_model", "eagle", "eagle3", "medusa", "mlp_speculator"})

VLLM_SERVE_ARGV_PREFIX = ("vllm", "serve")


@dataclass(frozen=True)
class VllmServerCommand:
    model: str
    host: str
    port: int
    dtype: str
    revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    speculative_method: str | None = None
    speculative_model: str | None = None
    num_speculative_tokens: int | None = None

    def speculative_config(self) -> dict[str, object] | None:
        """The --speculative-config payload, or None when speculative decoding is off."""
        if self.speculative_method is None:
            return None
        config: dict[str, object] = {"method": self.speculative_method}
        if self.speculative_model is not None:
            config["model"] = self.speculative_model
        config["num_speculative_tokens"] = self.num_speculative_tokens
        return config

    def argv(self) -> list[str]:
        args = [
            *VLLM_SERVE_ARGV_PREFIX,
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
        if self.revision:
            args.extend(["--revision", self.revision])
        if self.tokenizer:
            args.extend(["--tokenizer", self.tokenizer])
        if self.tokenizer_revision:
            args.extend(["--tokenizer-revision", self.tokenizer_revision])
        if self.max_model_len:
            args.extend(["--max-model-len", str(self.max_model_len)])
        if self.gpu_memory_utilization:
            args.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        # Both features default to True in vLLM, so an omitted flag would silently turn a
        # "no prefix caching" or "no chunked prefill" baseline into the opposite treatment.
        args.append(_boolean_flag("enable-prefix-caching", self.enable_prefix_caching))
        args.append(_boolean_flag("enable-chunked-prefill", self.enable_chunked_prefill))
        if self.max_num_batched_tokens:
            args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.max_num_seqs:
            args.extend(["--max-num-seqs", str(self.max_num_seqs)])
        speculative_config = self.speculative_config()
        if speculative_config is not None:
            args.extend(["--speculative-config", _speculative_config_json(speculative_config)])
        return args

    def shell_command(self) -> str:
        return " ".join(_shell_quote(part) for part in self.argv())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["argv"] = self.argv()
        payload["shell_command"] = self.shell_command()
        payload["speculative_config"] = self.speculative_config()
        return payload


def build_vllm_command(
    *,
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    dtype: str = "auto",
    revision: str | None = None,
    tokenizer: str | None = None,
    tokenizer_revision: str | None = None,
    quantization: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = None,
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: bool = False,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    speculative_method: str | None = None,
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
    if quantization is not None:
        normalize_vllm_quantization(quantization)
    resolved_method = _resolve_speculative_method(
        speculative_method=speculative_method,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
    )
    if revision is not None:
        require_immutable_revision(revision)
    resolved_tokenizer = tokenizer or model
    if tokenizer is None or tokenizer == model:
        resolved_tokenizer_revision = tokenizer_revision or revision
    else:
        resolved_tokenizer_revision = tokenizer_revision
    if resolved_tokenizer_revision is not None:
        require_immutable_revision(resolved_tokenizer_revision)
    normalized_dtype = normalize_vllm_dtype(dtype)
    return VllmServerCommand(
        model=model,
        host=host,
        port=port,
        dtype=normalized_dtype,
        revision=revision,
        tokenizer=resolved_tokenizer,
        tokenizer_revision=resolved_tokenizer_revision,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        speculative_method=resolved_method,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
    )


def _resolve_speculative_method(
    *,
    speculative_method: str | None,
    speculative_model: str | None,
    num_speculative_tokens: int | None,
) -> str | None:
    requested = speculative_method or speculative_model or num_speculative_tokens
    if requested is None:
        return None
    # A draft checkpoint without an explicit method is the documented draft-model setup.
    method = speculative_method or "draft_model"
    if not method.strip():
        raise ValueError("speculative_method must be a non-empty string")
    if method in VLLM_SPECULATIVE_METHODS_REQUIRING_MODEL and not speculative_model:
        raise ValueError(f"speculative_model is required for speculative method {method!r}")
    if not isinstance(num_speculative_tokens, int) or isinstance(num_speculative_tokens, bool) or num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be a positive integer for speculative decoding")
    return method


def normalize_vllm_dtype(dtype: str) -> str:
    normalized = DTYPE_ALIASES.get(dtype.lower(), dtype.lower())
    if normalized not in VLLM_DTYPES:
        allowed = ", ".join(sorted(VLLM_DTYPES))
        raise ValueError(f"unsupported vLLM dtype {dtype!r}; expected one of {allowed}")
    return normalized


def normalize_vllm_quantization(quantization: str) -> str:
    """Return the quantization value, rejecting names vLLM cannot resolve ("none" means unquantized)."""
    if quantization == "none":
        return quantization
    if quantization not in VLLM_QUANTIZATION_METHODS:
        allowed = ", ".join(sorted(VLLM_QUANTIZATION_METHODS))
        raise ValueError(
            f"unsupported vLLM quantization method {quantization!r}; expected 'none' or one of {allowed}"
        )
    return quantization


def require_immutable_revision(revision: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("revision must be a full 40 to 64 character lowercase hexadecimal identifier")
    return revision


def optimization_profile_name(
    *,
    enable_prefix_caching: bool,
    enable_chunked_prefill: bool,
    speculative: bool,
    quantization: str | None,
) -> str:
    features = []
    if enable_prefix_caching:
        features.append("prefix-cache")
    if enable_chunked_prefill:
        features.append("chunked-prefill")
    if speculative:
        features.append("speculative")
    if quantization and quantization != "none":
        features.append(f"quantized-{quantization}")
    return "+".join(features) if features else "baseline"


def is_vllm_serve_argv(argv: list[str] | tuple[str, ...]) -> bool:
    return tuple(argv[:2]) == VLLM_SERVE_ARGV_PREFIX and len(argv) > 2 and not argv[2].startswith("-")


def vllm_served_model(argv: list[str] | tuple[str, ...]) -> str | None:
    """The positional model argument of `vllm serve <model>`."""
    return argv[2] if is_vllm_serve_argv(argv) else None


def vllm_flag_value(argv: list[str] | tuple[str, ...], flag: str) -> str | None:
    try:
        index = list(argv).index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def vllm_boolean_flag(argv: list[str] | tuple[str, ...], name: str) -> bool | None:
    """True, False, or None when neither `--<name>` nor `--no-<name>` is present."""
    if f"--{name}" in argv:
        return None if f"--no-{name}" in argv else True
    return False if f"--no-{name}" in argv else None


def vllm_speculative_config(argv: list[str] | tuple[str, ...]) -> dict[str, object] | None:
    """The parsed --speculative-config payload, or None when absent or unparsable."""
    raw = vllm_flag_value(argv, "--speculative-config")
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _speculative_config_json(config: dict[str, object]) -> str:
    # Compact and insertion-ordered so the command text, and its SHA-256, stay reproducible.
    return json.dumps(config, separators=(",", ":"))


def _boolean_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def _shell_quote(value: str) -> str:
    if all(char.isalnum() or char in "._-/:=" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
