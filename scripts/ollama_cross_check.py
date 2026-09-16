"""Cross-validate this tool's client-side measurements against Ollama's own counters.

A benchmark client can only time bytes on a socket. Ollama additionally reports what its
runner actually did: `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, and
`eval_duration` on `/api/chat` and `/api/generate`. Running both against the same prompts
turns "our TTFT is 60 ms" into a checkable claim instead of an assertion.

The script is deliberately sequential. Concurrency would mix queueing into the native
durations and make the comparison meaningless.

    python scripts/ollama_cross_check.py \
        --model qwen2.5:1.5b-instruct \
        --prompts results/published/<study>/prompts_shared_prefix.jsonl \
        --output-dir results/published/<study>/cross-check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_accel.metrics.io import write_json, write_text_atomic  # noqa: E402
from llm_accel.metrics.manifest import write_run_manifest  # noqa: E402
from llm_accel.serving.openai_client import OpenAICompatibleClient  # noqa: E402
from llm_accel.workloads.prompts import load_prompt_file  # noqa: E402
from llm_accel.workloads.synthetic import prompt_batch  # noqa: E402


NANOSECONDS_PER_MS = 1_000_000


def _post(url: str, payload: dict[str, object], timeout_seconds: float) -> dict[str, object]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _native_call(
    server_root: str,
    route: str,
    payload: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    started = time.perf_counter()
    body = _post(f"{server_root}{route}", payload, timeout_seconds)
    wall_ms = (time.perf_counter() - started) * 1000
    eval_count = int(body.get("eval_count", 0))
    eval_duration_ms = float(body.get("eval_duration", 0)) / NANOSECONDS_PER_MS
    prompt_eval_duration_ms = float(body.get("prompt_eval_duration", 0)) / NANOSECONDS_PER_MS
    load_duration_ms = float(body.get("load_duration", 0)) / NANOSECONDS_PER_MS
    return {
        "route": route,
        "client_wall_ms": wall_ms,
        "done_reason": body.get("done_reason"),
        "prompt_eval_count": body.get("prompt_eval_count"),
        "prompt_eval_cached_count": body.get("prompt_eval_cached_count"),
        "prompt_eval_duration_ms": prompt_eval_duration_ms,
        "eval_count": eval_count,
        "eval_duration_ms": eval_duration_ms,
        "load_duration_ms": load_duration_ms,
        "total_duration_ms": float(body.get("total_duration", 0)) / NANOSECONDS_PER_MS,
        # Ollama's eval_duration covers every generated token, so this is ms per token, not TPOT.
        "ms_per_output_token": eval_duration_ms / eval_count if eval_count else None,
        "server_prefill_ms": load_duration_ms + prompt_eval_duration_ms,
    }


def _relative_difference(measured: float | None, reference: float | None) -> float | None:
    if measured is None or reference is None or reference == 0:
        return None
    return (measured - reference) / reference


def _client_measurement(
    prompt: str,
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, object]:
    client = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        backend="ollama",
        request_timeout_seconds=timeout_seconds,
    )
    result = client.complete(prompt, max_tokens, stream=True)
    return {
        "output_tokens": result.output_tokens,
        "input_tokens": result.input_tokens,
        "token_count_method": result.token_count_method,
        "ttft_ms": result.ttft_ms,
        "total_latency_ms": result.total_latency_ms,
        "tpot_ms": result.tpot_ms,
        "inter_token_latency_sample_count": len(result.inter_token_latencies_ms),
        "inter_token_latency_p50_ms": (
            statistics.median(result.inter_token_latencies_ms)
            if result.inter_token_latencies_ms
            else None
        ),
        # Directly comparable to eval_duration / eval_count, which also includes the first token.
        "ms_per_output_token": (
            (result.total_latency_ms - result.ttft_ms) / (result.output_tokens - 1)
            if result.output_tokens > 1
            else None
        ),
    }


def collect_comparisons(
    prompts: list[str],
    *,
    server_root: str,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    """Measure each route over the whole prompt list before starting the next route.

    Ollama keeps served prompts in its runner cache, so only the route that sees a prompt first
    measures a real prefill; every later route gets a cache hit. That cannot be avoided when the
    routes must answer the same prompt, so the order is fixed and declared instead of hidden:
    `/api/chat` runs cold, and the OpenAI client runs warm.

    Token counts and per-token decode cost are unaffected by the cache, so those comparisons hold
    in either state. Prefill timings are reported as absolute numbers with their cache state
    rather than as an agreement ratio between a cold and a warm measurement.
    """
    options = {"temperature": 0, "num_predict": max_tokens}
    chat_rows = [
        _native_call(
            server_root,
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            },
            timeout_seconds,
        )
        for prompt in prompts
    ]
    generate_rows = [
        _native_call(
            server_root,
            "/api/generate",
            {"model": model, "prompt": prompt, "stream": False, "options": options},
            timeout_seconds,
        )
        for prompt in prompts
    ]
    client_rows = [
        _client_measurement(
            prompt,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        for prompt in prompts
    ]
    comparisons = []
    for prompt, client_view, native_chat, native_generate in zip(
        prompts, client_rows, chat_rows, generate_rows
    ):
        comparisons.append(
            {
                # Prompt text stays out of artifacts; the digest lines rows up across runs.
                "prompt_sha256_prefix": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                "client": {**client_view, "cache_state": "warm"},
                "native_chat": {**native_chat, "cache_state": "cold"},
                "native_generate": {**native_generate, "cache_state": "warm"},
                "agreement": {
                    "output_tokens_match_native_chat": client_view["output_tokens"] == native_chat["eval_count"],
                    "input_tokens_match_native_chat": client_view["input_tokens"] == native_chat["prompt_eval_count"],
                    "decode_ms_per_token_relative_difference": _relative_difference(
                        client_view["ms_per_output_token"], native_chat["ms_per_output_token"]
                    ),
                    "total_latency_relative_difference": _relative_difference(
                        client_view["total_latency_ms"], native_chat["total_duration_ms"]
                    ),
                },
            }
        )
    return comparisons


def _summarize(comparisons: list[dict[str, object]]) -> dict[str, object]:
    def median_agreement(key: str) -> float | None:
        values = [
            float(item["agreement"][key])
            for item in comparisons
            if item["agreement"].get(key) is not None
        ]
        return statistics.median(values) if values else None

    def median_field(route: str, key: str) -> float | None:
        values = [
            float(item[route][key]) for item in comparisons if item[route].get(key) is not None
        ]
        return statistics.median(values) if values else None

    return {
        "prompt_count": len(comparisons),
        "output_tokens_match_rate": sum(
            1 for item in comparisons if item["agreement"]["output_tokens_match_native_chat"]
        )
        / len(comparisons),
        "input_tokens_match_rate": sum(
            1 for item in comparisons if item["agreement"]["input_tokens_match_native_chat"]
        )
        / len(comparisons),
        "median_decode_ms_per_token_relative_difference": median_agreement(
            "decode_ms_per_token_relative_difference"
        ),
        "median_total_latency_relative_difference": median_agreement(
            "total_latency_relative_difference"
        ),
        "median_client_ms_per_output_token": median_field("client", "ms_per_output_token"),
        "median_native_chat_ms_per_output_token": median_field("native_chat", "ms_per_output_token"),
        "median_client_warm_ttft_ms": median_field("client", "ttft_ms"),
        "median_native_chat_cold_prefill_ms": median_field("native_chat", "server_prefill_ms"),
        "median_native_chat_cold_prompt_eval_cached_count": median_field(
            "native_chat", "prompt_eval_cached_count"
        ),
        "median_native_chat_prompt_eval_count": median_field("native_chat", "prompt_eval_count"),
        "median_native_generate_warm_prompt_eval_cached_count": median_field(
            "native_generate", "prompt_eval_cached_count"
        ),
    }


def _render_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    lines = [
        "# Ollama Native Cross-Check",
        "",
        f"- Model: `{payload['model']}`",
        f"- Prompts: `{summary['prompt_count']}`",
        f"- Requested output tokens: `{payload['max_tokens']}`",
        f"- Workload: `{payload['workload']}`",
        "",
        "| Check | Value |",
        "| --- | ---: |",
        f"| Output tokens equal to `eval_count` | {summary['output_tokens_match_rate']:.3f} |",
        f"| Input tokens equal to `prompt_eval_count` | {summary['input_tokens_match_rate']:.3f} |",
        f"| Median decode ms/token relative difference | {_format(summary['median_decode_ms_per_token_relative_difference'])} |",
        f"| Median total latency relative difference | {_format(summary['median_total_latency_relative_difference'])} |",
        f"| Client median ms per output token | {_ms(summary['median_client_ms_per_output_token'])} |",
        f"| Native `eval_duration / eval_count` median | {_ms(summary['median_native_chat_ms_per_output_token'])} |",
        f"| Native cold prefill median (`load` + `prompt_eval`) | {_ms(summary['median_native_chat_cold_prefill_ms'])} |",
        f"| Client warm TTFT median | {_ms(summary['median_client_warm_ttft_ms'])} |",
        f"| Median `prompt_eval_count` | {_count(summary['median_native_chat_prompt_eval_count'])} |",
        f"| Median cold `prompt_eval_cached_count` (`/api/chat`) | {_count(summary['median_native_chat_cold_prompt_eval_cached_count'])} |",
        f"| Median warm `prompt_eval_cached_count` (`/api/generate`) | {_count(summary['median_native_generate_warm_prompt_eval_cached_count'])} |",
        "",
        "Per-prompt records are in `cross_check.json`.",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in payload["notes"]],
        "",
    ]
    return "\n".join(lines)


def _format(value: object) -> str:
    return f"{float(value):+.4f}" if isinstance(value, (int, float)) else "n/a"


def _ms(value: object) -> str:
    return f"{float(value):.2f} ms" if isinstance(value, (int, float)) else "n/a"


def _count(value: object) -> str:
    return f"{float(value):.1f}" if isinstance(value, (int, float)) else "n/a"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", help="Prompt file; omit to use the same synthetic workload as the benchmark")
    parser.add_argument("--synthetic-count", type=int, default=8)
    parser.add_argument("--synthetic-input-tokens", type=int, default=128)
    parser.add_argument("--synthetic-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    if args.prompts:
        prompts = load_prompt_file(args.prompts)[: args.synthetic_count]
        workload = f"prompt file {Path(args.prompts).name}"
    else:
        prompts = prompt_batch(args.synthetic_count, args.synthetic_input_tokens, args.synthetic_seed)
        workload = (
            f"synthetic seed={args.synthetic_seed} input_tokens={args.synthetic_input_tokens}"
        )
    base_url = f"{args.server_root.rstrip('/')}/v1"
    comparisons = collect_comparisons(
        prompts,
        server_root=args.server_root.rstrip("/"),
        base_url=base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    payload = {
        "schema_version": "0.2",
        "server_root": args.server_root,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "workload": workload,
        "comparisons": comparisons,
        "summary": _summarize(comparisons),
        "notes": [
            "Routes run one full pass at a time in a fixed order: /api/chat cold, then /api/generate warm, then the OpenAI client warm.",
            "Ollama caches served prompts, so only the first route to see a prompt measures a real prefill; the cache_state field on each route records which state it was in.",
            "Token counts and per-token decode cost do not depend on the cache, so those comparisons are valid across routes.",
            "Ollama reports eval_duration over all generated tokens, so the client figure compared against it is the decode span divided by output tokens minus one.",
            "Prefill is reported as absolute medians with their cache state rather than as an agreement ratio between a cold and a warm measurement.",
            "/api/generate applies the model's chat template with its default system message, so its prompt token count need not equal the /v1 chat count.",
            "prompt_eval_cached_count shows how much of each prompt the runner served from its resident prefix rather than re-computing.",
        ],
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "cross_check.json", payload)
    write_text_atomic(output_dir / "cross_check.md", _render_markdown(payload))
    write_run_manifest(
        output_dir,
        run_type="ollama_cross_check",
        artifacts=["manifest.json", "cross_check.json", "cross_check.md"],
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
