from __future__ import annotations

import json
from pathlib import Path

from llm_accel.metrics.io import write_json
from llm_accel.metrics.manifest import write_run_manifest
from llm_accel.serving.openai_client import OpenAICompatibleClient


def load_task_specs(path: str | Path) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict) or not isinstance(payload.get("prompt"), str):
            raise ValueError(f"line {line_no} must contain a prompt string")
        keywords = payload.get("expected_keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(item, str) for item in keywords):
            raise ValueError(f"line {line_no} expected_keywords must be a list of strings")
        specs.append({"prompt": payload["prompt"], "expected_keywords": keywords})
    if not specs:
        raise ValueError("task spec file must contain at least one prompt")
    return specs


def evaluate_tasks(
    *,
    base_url: str,
    model: str,
    task_specs: list[dict[str, object]],
    output_dir: str | Path,
    backend: str = "openai-compatible",
    max_tokens: int = 64,
    stream: bool = False,
) -> dict[str, object]:
    client = OpenAICompatibleClient(base_url=base_url, model=model, backend=backend)
    checks = []
    for index, spec in enumerate(task_specs):
        prompt = str(spec["prompt"])
        expected_keywords = [str(item).lower() for item in spec.get("expected_keywords", [])]
        try:
            result = client.complete(prompt, max_tokens=max_tokens, request_index=index, stream=stream)
            output = result.output_text.lower()
            matched = [keyword for keyword in expected_keywords if keyword.lower() in output]
            score = len(matched) / len(expected_keywords) if expected_keywords else 1.0
            checks.append(
                {
                    "prompt_index": index,
                    "expected_keywords": expected_keywords,
                    "matched_keywords": matched,
                    "keyword_score": score,
                    "output_tokens": result.output_tokens,
                    "total_latency_ms": result.total_latency_ms,
                    "error": None,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "prompt_index": index,
                    "expected_keywords": expected_keywords,
                    "matched_keywords": [],
                    "keyword_score": 0.0,
                    "output_tokens": 0,
                    "total_latency_ms": 0.0,
                    "error": str(exc),
                }
            )

    mean_score = sum(float(check["keyword_score"]) for check in checks) / len(checks)
    report = {
        "model": model,
        "backend": "mock" if base_url.startswith("mock://") else backend,
        "base_url": base_url if base_url.startswith(("mock://", "http://localhost", "http://127.0.0.1")) else "redacted",
        "task_count": len(task_specs),
        "mean_keyword_score": mean_score,
        "passed": mean_score == 1.0 and all(check["error"] is None for check in checks),
        "checks": checks,
        "notes": [
            "Keyword scoring is a lightweight task-specific sanity check.",
            "It should be replaced or extended for domain-specific quality evaluation.",
        ],
    }
    out_dir = Path(output_dir)
    write_json(out_dir / "task_eval.json", report)
    _write_markdown(out_dir / "task_eval.md", report)
    write_run_manifest(
        out_dir,
        run_type="task_quality_eval",
        artifacts=["manifest.json", "task_eval.json", "task_eval.md"],
    )
    return report


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    rows = [
        f"| {check['prompt_index']} | {check['keyword_score']:.3f} | "
        f"`{', '.join(check['matched_keywords'])}` | `{check['error']}` |"
        for check in report["checks"]  # type: ignore[index]
    ]
    text = "\n".join(
        [
            "# Task Quality Evaluation",
            "",
            f"- Model: `{report['model']}`",
            f"- Backend: `{report['backend']}`",
            f"- Mean keyword score: `{report['mean_keyword_score']:.3f}`",
            f"- Passed: `{report['passed']}`",
            "",
            "| Prompt | Keyword score | Matched keywords | Error |",
            "| ---: | ---: | --- | --- |",
            *rows,
            "",
            "Keyword scoring is a lightweight sanity check, not a full quality benchmark.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
