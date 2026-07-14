from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
            if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
                self.wfile.write(b'data: {"choices":[{"text":"hello "}]}\n\n')
            else:
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
                )
            self.wfile.flush()
            time.sleep(0.01)
            if self.path.endswith("/completions") and not self.path.endswith("/chat/completions"):
                self.wfile.write(b'data: {"choices":[{"text":"world"}]}\n\n')
            else:
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
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
