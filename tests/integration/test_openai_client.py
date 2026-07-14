from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from llm_accel.benchmarks.latency import run_latency_benchmark
from llm_accel.serving.openai_client import OpenAICompatibleClient


class _OpenAIHandler(BaseHTTPRequestHandler):
    seen_paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        self.seen_paths.append(self.path)
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            prompt = payload.get("prompt") or payload.get("messages", [{}])[0].get("content")
            if prompt == "测试":
                chunks = ["你好", "世界"]
            else:
                chunks = ["hello ", "world"]
            if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
                self.wfile.write(
                    f'data: {{"choices":[{{"text":{json.dumps(chunks[0])}}}]}}\n\n'.encode()
                )
            else:
                self.wfile.write(
                    f'data: {{"choices":[{{"delta":{{"content":{json.dumps(chunks[0])}}}}}]}}\n\n'.encode()
                )
            self.wfile.flush()
            time.sleep(0.01)
            if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
                self.wfile.write(
                    f'data: {{"choices":[{{"text":{json.dumps(chunks[1])}}}]}}\n\n'.encode()
                )
            else:
                self.wfile.write(
                    f'data: {{"choices":[{{"delta":{{"content":{json.dumps(chunks[1])}}}}}]}}\n\n'.encode()
                )
            if payload.get("stream_options") == {"include_usage": True}:
                self.wfile.write(
                    b'data: {"choices":[],"usage":{"prompt_tokens":17,"completion_tokens":4}}\n\n'
                )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
            body = {
                "choices": [{"text": "hello world"}],
                "usage": {"completion_tokens": 2},
            }
        else:
            body = {
                "choices": [{"message": {"content": "hello world"}}],
                "usage": {"completion_tokens": 2},
            }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_server() -> tuple[ThreadingHTTPServer, str]:
    _OpenAIHandler.seen_paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


def test_openai_client_non_streaming() -> None:
    server, base_url = _start_server()
    try:
        client = OpenAICompatibleClient(base_url=base_url, model="mock")
        result = client.complete("hello", max_tokens=2, stream=False)
    finally:
        server.shutdown()

    assert result.output_text == "hello world"
    assert result.output_tokens == 2
    assert result.ttft_ms == result.total_latency_ms
    assert _OpenAIHandler.seen_paths == ["/v1/chat/completions"]


def test_openai_client_streaming_observes_ttft() -> None:
    server, base_url = _start_server()
    try:
        client = OpenAICompatibleClient(base_url=base_url, model="mock")
        result = client.complete("hello", max_tokens=2, stream=True)
    finally:
        server.shutdown()

    assert result.output_text == "hello world"
    assert result.output_tokens == 2
    assert result.ttft_ms < result.total_latency_ms
    assert _OpenAIHandler.seen_paths == ["/v1/chat/completions"]


def test_vllm_streaming_counts_final_text_with_resolved_tokenizer() -> None:
    class CharacterTokenCounter:
        method = "tokenizers.encode(add_special_tokens=false)"

        def count(self, text: str) -> int:
            return len(text)

    server, base_url = _start_server()
    try:
        client = OpenAICompatibleClient(
            base_url=base_url,
            model="mock",
            backend="vllm",
            token_counter=CharacterTokenCounter(),
        )
        result = client.complete("测试", max_tokens=8, stream=True)
    finally:
        server.shutdown()

    assert result.output_text == "你好世界"
    assert result.input_tokens == 17
    assert result.output_tokens == 4
    assert result.token_count_method == (
        "prompt=server_usage;output=tokenizers.encode(add_special_tokens=false)"
    )


def test_vllm_benchmark_binds_tokenizer_counts_and_method(monkeypatch, tmp_path) -> None:
    class CharacterTokenCounter:
        method = "tokenizers.encode(add_special_tokens=false)"

        def count(self, text: str) -> int:
            return len(text)

    monkeypatch.setattr(
        "llm_accel.benchmarks.latency.load_token_counter",
        lambda tokenizer, revision: CharacterTokenCounter(),
    )
    monkeypatch.setattr(
        "llm_accel.serving.openai_client.load_token_counter",
        lambda tokenizer, revision: CharacterTokenCounter(),
    )
    server, base_url = _start_server()
    try:
        summary = run_latency_benchmark(
            base_url=base_url,
            model="mock",
            backend="vllm",
            tokenizer="resolved-tokenizer",
            tokenizer_revision="b" * 40,
            concurrency=1,
            input_tokens=2,
            output_tokens=8,
            output_dir=tmp_path,
            request_count=1,
            prompt_texts=["测试"],
        )
    finally:
        server.shutdown()

    row = json.loads((tmp_path / "raw_requests.jsonl").read_text(encoding="utf-8"))
    assert row["input_tokens"] == 17
    assert row["output_tokens"] == 4
    assert row["token_count_method"] == (
        "prompt=server_usage;output=tokenizers.encode(add_special_tokens=false)"
    )
    assert summary["metadata"]["token_count_method"] == (
        "prompt=server_usage;output=tokenizers.encode(add_special_tokens=false)"
    )
    assert summary["metrics"]["output_tokens"] == 4


def test_vllm_benchmark_excludes_deferred_tokenization_from_measurement(
    monkeypatch,
    tmp_path,
) -> None:
    class SlowCharacterTokenCounter:
        method = "tokenizers.encode(add_special_tokens=false)"

        def count(self, text: str) -> int:
            time.sleep(0.15)
            return len(text)

    monkeypatch.setattr(
        "llm_accel.benchmarks.latency.load_token_counter",
        lambda tokenizer, revision: SlowCharacterTokenCounter(),
    )
    monkeypatch.setattr(
        "llm_accel.serving.openai_client.load_token_counter",
        lambda tokenizer, revision: SlowCharacterTokenCounter(),
    )
    server, base_url = _start_server()
    try:
        summary = run_latency_benchmark(
            base_url=base_url,
            model="mock",
            backend="vllm",
            tokenizer="resolved-tokenizer",
            tokenizer_revision="b" * 40,
            concurrency=1,
            input_tokens=2,
            output_tokens=8,
            output_dir=tmp_path,
            request_count=2,
            prompt_texts=["测试"],
        )
    finally:
        server.shutdown()

    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    measured = summary["metrics"]["throughput"]["measured_elapsed_seconds"]
    assert measured < 0.1
    assert max(row["total_latency_ms"] for row in rows) < 100.0
    assert rows[1]["dispatch_offset_ms"] - rows[0]["completed_offset_ms"] < 50.0


def test_vllm_benchmark_deferred_clients_do_not_reload_tokenizer(
    monkeypatch,
    tmp_path,
) -> None:
    class CharacterTokenCounter:
        method = "tokenizers.encode(add_special_tokens=false)"

        def count(self, text: str) -> int:
            return len(text)

    monkeypatch.setattr(
        "llm_accel.benchmarks.latency.load_token_counter",
        lambda tokenizer, revision: CharacterTokenCounter(),
    )

    def reject_worker_load(tokenizer, revision):
        raise AssertionError("measured clients must not load the tokenizer")

    monkeypatch.setattr(
        "llm_accel.serving.openai_client.load_token_counter",
        reject_worker_load,
    )
    server, base_url = _start_server()
    try:
        summary = run_latency_benchmark(
            base_url=base_url,
            model="mock",
            backend="vllm",
            tokenizer="resolved-tokenizer",
            tokenizer_revision="b" * 40,
            concurrency=1,
            input_tokens=2,
            output_tokens=8,
            output_dir=tmp_path,
            request_count=2,
            prompt_texts=["测试"],
        )
    finally:
        server.shutdown()

    assert summary["metrics"]["completed_count"] == 2
    assert summary["metrics"]["failed_count"] == 0


def test_multiprocess_vllm_workers_do_not_resolve_tokenizer_in_measurement(
    monkeypatch,
    tmp_path,
) -> None:
    class CharacterTokenCounter:
        method = "tokenizers.encode(add_special_tokens=false)"

        def count(self, text: str) -> int:
            return len(text)

    monkeypatch.setattr(
        "llm_accel.benchmarks.latency.load_token_counter",
        lambda tokenizer, revision: CharacterTokenCounter(),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    server, base_url = _start_server()
    try:
        summary = run_latency_benchmark(
            base_url=base_url,
            model="mock",
            backend="vllm",
            tokenizer="not-cached/tokenizer",
            tokenizer_revision="b" * 40,
            concurrency=2,
            input_tokens=2,
            output_tokens=8,
            output_dir=tmp_path,
            request_count=4,
            prompt_texts=["测试"],
            client_processes=2,
        )
    finally:
        server.shutdown()

    rows = [
        json.loads(line)
        for line in (tmp_path / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["metrics"]["completed_count"] == 4
    assert summary["metrics"]["failed_count"] == 0
    assert all(row["input_tokens"] == 17 for row in rows)
    assert all(row["output_tokens"] == 4 for row in rows)


def test_vllm_benchmark_rejects_mutable_local_tokenizer_evidence(tmp_path) -> None:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    server, base_url = _start_server()
    try:
        with pytest.raises(ValueError, match="local tokenizer"):
            run_latency_benchmark(
                base_url=base_url,
                model="mock",
                backend="vllm",
                tokenizer=str(tokenizer_path),
                tokenizer_revision="b" * 40,
                concurrency=1,
                input_tokens=2,
                output_tokens=8,
                output_dir=tmp_path / "run",
                request_count=1,
                prompt_texts=["测试"],
            )
    finally:
        server.shutdown()


def test_openai_client_completion_endpoint_non_streaming() -> None:
    server, base_url = _start_server()
    try:
        client = OpenAICompatibleClient(base_url=base_url, model="mock", api_kind="completion")
        result = client.complete("hello", max_tokens=2, stream=False)
    finally:
        server.shutdown()

    assert result.output_text == "hello world"
    assert result.output_tokens == 2
    assert _OpenAIHandler.seen_paths == ["/v1/completions"]


def test_openai_client_completion_endpoint_streaming() -> None:
    server, base_url = _start_server()
    try:
        client = OpenAICompatibleClient(base_url=base_url, model="mock", api_kind="completion")
        result = client.complete("hello", max_tokens=2, stream=True)
    finally:
        server.shutdown()

    assert result.output_text == "hello world"
    assert result.output_tokens == 2
    assert result.ttft_ms < result.total_latency_ms
    assert _OpenAIHandler.seen_paths == ["/v1/completions"]


def test_open_loop_delayed_endpoint_exposes_client_queue_saturation(tmp_path) -> None:
    server, base_url = _start_server()
    try:
        summary = run_latency_benchmark(
            base_url=base_url,
            model="mock-model",
            backend="openai-compatible",
            concurrency=1,
            input_tokens=8,
            output_tokens=2,
            output_dir=tmp_path / "delayed-open-loop",
            request_count=4,
            request_schedule="open-loop",
            request_rate_rps=1000.0,
            queue_delay_warning_ms=1.0,
        )
    finally:
        server.shutdown()

    assert summary["metrics"]["queue_delay_ms"]["p95"] > 1.0
    assert any("Client saturation detected" in warning for warning in summary["warnings"])
    assert summary["metrics"]["end_to_end_latency_ms"]["p95"] > summary["metrics"]["latency_ms"]["p95"]
