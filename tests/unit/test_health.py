from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_accel.serving.health import check_endpoint_health


def test_mock_endpoint_health_is_available() -> None:
    health = check_endpoint_health("mock://local")

    assert health["healthy"] is True
    assert health["status"] == "healthy"
    assert health["error"] is None


class _ModelsHandler(BaseHTTPRequestHandler):
    status_code = 200
    body = b'{"object":"list","data":[]}'

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(self.status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format: str, *args: object) -> None:
        return


def _serve(status_code: int, body: bytes) -> tuple[ThreadingHTTPServer, str]:
    handler = type("Handler", (_ModelsHandler,), {"status_code": status_code, "body": body})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    return server, f"http://{host}:{port}/v1"


@pytest.mark.parametrize(
    ("status_code", "body", "expected_status", "healthy"),
    [
        (200, b'{"object":"list","data":[]}', "healthy", True),
        (401, b'{"error":"unauthorized"}', "unauthorized", False),
        (403, b'{"error":"forbidden"}', "unauthorized", False),
        (404, b'{"error":"not found"}', "unhealthy", False),
        (500, b'{"error":"boom"}', "unhealthy", False),
        (200, b"not json", "unhealthy", False),
    ],
)
def test_endpoint_health_distinguishes_http_outcomes(status_code, body, expected_status, healthy) -> None:
    server, base_url = _serve(status_code, body)
    try:
        health = check_endpoint_health(base_url, timeout_seconds=2.0)
    finally:
        server.shutdown()

    assert health["status"] == expected_status
    assert health["healthy"] is healthy
    assert health["status_code"] == status_code
    assert (health["error"] is None) is healthy


def test_endpoint_health_reports_unreachable_endpoint() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    health = check_endpoint_health(f"http://127.0.0.1:{port}/v1", timeout_seconds=1.0)

    assert health["status"] == "unreachable"
    assert health["healthy"] is False
    assert health["status_code"] is None
