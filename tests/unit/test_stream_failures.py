"""A benchmark that counts a failed response as a success publishes a false number."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llm_accel.cli import main
from llm_accel.serving.openai_client import OpenAICompatibleClient


def _delta(text: str) -> bytes:
    event = {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
    return b"data: " + json.dumps(event).encode() + b"\n\n"


STOP = b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
DONE = b"data: [DONE]\n\n"

FAILED_STREAMS = {
    "openai_error_object": b'data: {"error": {"message": "backend overloaded"}}\n\n' + DONE,
    "error_after_tokens": _delta("partial ") + b'data: {"error": "backend overloaded"}\n\n' + DONE,
    "vllm_error_object": b'data: {"object": "error", "message": "backend overloaded", "code": 500}\n\n' + DONE,
    "error_finish_reason": _delta("partial ") + b'data: {"choices": [{"delta": {}, "finish_reason": "error"}]}\n\n' + DONE,
    "truncated_without_termination": _delta("partial "),
    "aborted_after_tokens": _delta("partial ") + b'data: {"choices": [{"delta": {}, "finish_reason": "abort"}]}\n\n' + DONE,
    "terminated_without_tokens": DONE,
}

FAILED_RESPONSES = {
    "error_body": {"error": {"message": "backend overloaded"}},
    "error_object": {"object": "error", "message": "backend overloaded"},
    "aborted": {"choices": [{"message": {"content": "partial"}, "finish_reason": "abort"}]},
    "null_content": {"choices": [{"message": {"content": None}}], "usage": {"prompt_tokens": 1, "completion_tokens": 0}},
    "empty_content_without_usage": {"choices": [{"message": {"content": ""}}]},
}

COMPLETE_STREAMS = {
    "finish_reason_and_done": _delta("one ") + _delta("two") + STOP + DONE,
    "done_only": _delta("one ") + _delta("two") + DONE,
    "finish_reason_only": _delta("one ") + _delta("two") + STOP,
}


@pytest.fixture
def stream_endpoint():
    bodies: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(bodies[0])

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def serve(body: bytes) -> str:
        bodies[:] = [body]
        return f"http://127.0.0.1:{server.server_port}/v1"

    yield serve
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("body", FAILED_STREAMS.values(), ids=FAILED_STREAMS.keys())
def test_client_raises_on_a_failed_stream(stream_endpoint, body: bytes) -> None:
    client = OpenAICompatibleClient(stream_endpoint(body), "m")

    with pytest.raises(RuntimeError):
        client.complete("hello", 8)


@pytest.mark.parametrize("body", COMPLETE_STREAMS.values(), ids=COMPLETE_STREAMS.keys())
def test_client_accepts_a_complete_stream(stream_endpoint, body: bytes) -> None:
    result = OpenAICompatibleClient(stream_endpoint(body), "m").complete("hello", 8)

    assert result.output_text == "one two"


@pytest.mark.parametrize("body", FAILED_RESPONSES.values(), ids=FAILED_RESPONSES.keys())
def test_client_raises_on_a_failed_non_streaming_response(stream_endpoint, body: dict) -> None:
    client = OpenAICompatibleClient(stream_endpoint(json.dumps(body).encode()), "m")

    with pytest.raises(RuntimeError):
        client.complete("hello", 8, stream=False)


def test_client_accepts_a_complete_non_streaming_response(stream_endpoint) -> None:
    body = {"choices": [{"message": {"content": "one two"}, "finish_reason": "stop"}]}

    result = OpenAICompatibleClient(stream_endpoint(json.dumps(body).encode()), "m").complete("hello", 8, stream=False)

    assert result.output_text == "one two"


def test_failed_streams_are_counted_as_failures_in_the_summary(stream_endpoint, tmp_path: Path) -> None:
    code = main(
        [
            "bench",
            "throughput",
            "--base-url",
            stream_endpoint(FAILED_STREAMS["openai_error_object"]),
            "--model",
            "m",
            "--concurrency",
            "1",
            "--output-tokens",
            "8",
            "--request-count",
            "2",
            "--slo-ttft-ms",
            "1000",
            "--slo-tpot-ms",
            "15",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )

    assert code in {0, 1}
    summary = json.loads(next((tmp_path / "run").rglob("summary.json")).read_text(encoding="utf-8"))
    counts = json.dumps(summary)
    assert '"failed_count": 2' in counts
    assert '"completed_count": 0' in counts
    rows = [json.loads(line) for line in next((tmp_path / "run").rglob("raw_requests.jsonl")).open()]
    assert all(row["completed"] is False and "backend overloaded" in row["error"] for row in rows)
