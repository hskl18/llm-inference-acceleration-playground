from __future__ import annotations

from pathlib import Path

from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.serving.openai_client import OpenAICompatibleClient


def evaluate_prompts(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    output_dir: str | Path,
    backend: str = "openai-compatible",
    max_tokens: int = 64,
    stream: bool = False,
) -> dict[str, object]:
    if not prompts:
        raise ValueError("prompts must not be empty")

    client = OpenAICompatibleClient(base_url=base_url, model=model, backend=backend)
    checks = []
    for index, prompt in enumerate(prompts):
        try:
            result = client.complete(prompt, max_tokens=max_tokens, request_index=index, stream=stream)
            output = result.output_text.strip()
            checks.append(
                {
                    "prompt_index": index,
                    "prompt_chars": len(prompt),
                    "non_empty": bool(output),
                    "output_chars": len(output),
                    "output_tokens": result.output_tokens,
                    "ttft_ms": result.ttft_ms,
                    "total_latency_ms": result.total_latency_ms,
                    "error": None,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "prompt_index": index,
                    "prompt_chars": len(prompt),
                    "non_empty": False,
                    "output_chars": 0,
                    "output_tokens": 0,
                    "ttft_ms": 0.0,
                    "total_latency_ms": 0.0,
                    "error": str(exc),
                }
            )

    passed = all(check["non_empty"] and check["error"] is None for check in checks)
    report = {
        "model": model,
        "backend": "mock" if base_url.startswith("mock://") else backend,
        "base_url": base_url if base_url.startswith(("mock://", "http://localhost", "http://127.0.0.1")) else "redacted",
        "prompt_count": len(prompts),
        "passed": passed,
        "checks": checks,
        "notes": [
            "This is a lightweight sanity evaluation, not a task-specific quality benchmark.",
            "It checks output presence, output length, errors, and latency metadata.",
        ],
    }
    out_dir = Path(output_dir)
    write_json(out_dir / "quality_eval.json", report)
    _write_markdown(out_dir / "quality_eval.md", report)
    write_run_manifest(
        out_dir,
        run_type="quality_sanity_eval",
        artifacts=[
            "manifest.json",
            "quality_eval.json",
            "quality_eval.md",
        ],
    )
    return report


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = [
        f"| {check['prompt_index']} | {check['non_empty']} | {check['output_tokens']} | "
        f"{check['total_latency_ms']:.3f} | `{check['error']}` |"
        for check in report["checks"]  # type: ignore[index]
    ]
    text = "\n".join(
        [
            "# Quality Sanity Evaluation",
            "",
            f"- Model: `{report['model']}`",
            f"- Backend: `{report['backend']}`",
            f"- Passed: `{report['passed']}`",
            "",
            "| Prompt | Non-empty | Output tokens | Total latency ms | Error |",
            "| ---: | --- | ---: | ---: | --- |",
            *rows,
            "",
            "This is a lightweight sanity evaluation, not a substitute for domain-specific quality evaluation.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
