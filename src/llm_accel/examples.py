from __future__ import annotations

from importlib import resources
from pathlib import Path

from llm_accel.metrics.io import write_text_atomic


EXAMPLE_FILENAMES = (
    "benchmark_small.yaml",
    "benchmark_local.yaml",
    "benchmark_prompts.yaml",
    "benchmark_prefix_cache.yaml",
    "benchmark_vllm_small.yaml",
    "spec_prompts.jsonl",
    "prefix_cache_prompts.jsonl",
    "task_eval_small.jsonl",
)


def list_example_files() -> list[str]:
    return list(EXAMPLE_FILENAMES)


def write_example_files(output_dir: str | Path, *, overwrite: bool = False) -> list[str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for filename in EXAMPLE_FILENAMES:
        target = destination / filename
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} already exists; pass --overwrite to replace it")
        source = resources.files("llm_accel").joinpath("example_configs", filename)
        write_text_atomic(target, source.read_text(encoding="utf-8"))
        written.append(str(target))

    return written
