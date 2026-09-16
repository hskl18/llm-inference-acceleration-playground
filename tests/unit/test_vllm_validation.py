import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_accel.serving.vllm_validation import validate_vllm_environment


class _FakeVllm(BaseHTTPRequestHandler):
    """Minimal stand-in for a vLLM server: /v1/models, /version, and Prometheus /metrics."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": "test-model"}]}).encode("utf-8"))
        elif self.path == "/version":
            self._send(200, json.dumps({"version": "0.29.0"}).encode("utf-8"))
        elif self.path == "/metrics":
            self._send(200, b'vllm:prefix_cache_queries_total{model_name="test-model"} 7.0\n')
        else:
            self._send(404, b"{}")

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _Unauthorized(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(401)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def fake_vllm():
    server = _serve(_FakeVllm)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


def test_validate_vllm_environment_writes_report(tmp_path) -> None:
    report = validate_vllm_environment(
        model="test-model",
        base_url="mock://local",
        output_dir=tmp_path,
        enable_chunked_prefill=True,
    )

    assert "ready_for_hardware_benchmark" in report
    assert "--enable-chunked-prefill" in report["command"]["argv"]
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "vllm_validation.json").exists()
    assert (tmp_path / "vllm_validation.md").exists()


def test_validate_vllm_environment_reports_unauthorized_endpoint(tmp_path) -> None:
    server = _serve(_Unauthorized)
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        report = validate_vllm_environment(model="test-model", base_url=base_url, output_dir=tmp_path)
    finally:
        server.shutdown()

    assert report["checks"]["endpoint_health"]["status"] == "unauthorized"
    assert any(blocker.startswith("endpoint health check unauthorized") for blocker in report["blockers"])


def test_remote_validation_passes_without_local_vllm_or_gpu(tmp_path, fake_vllm: str) -> None:
    report = validate_vllm_environment(model="test-model", base_url=fake_vllm, output_dir=tmp_path)

    # This machine has no vLLM package and no NVIDIA GPU, yet the endpoint is fully described.
    assert report["checks"]["backend_version"] == "0.29.0"
    assert report["checks"]["backend_version_source"] == "server_version_endpoint"
    assert report["checks"]["serving_state"]["available"] is True
    assert report["same_host"] is False
    assert report["ready_for_hardware_benchmark"] is True
    assert report["blockers"] == []
    assert any("not declared same-host" in warning for warning in report["warnings"])


def test_same_host_run_still_requires_local_vllm_and_gpu(tmp_path, fake_vllm: str) -> None:
    report = validate_vllm_environment(
        model="test-model",
        base_url=fake_vllm,
        output_dir=tmp_path,
        same_host=True,
    )

    assert report["ready_for_hardware_benchmark"] is False
    assert any("not importable on this client" in blocker for blocker in report["blockers"])


def test_unreadable_server_version_blocks_a_hardware_claim(tmp_path) -> None:
    report = validate_vllm_environment(
        model="test-model",
        base_url="http://127.0.0.1:1/v1",
        output_dir=tmp_path,
        timeout_seconds=0.5,
    )

    assert report["ready_for_hardware_benchmark"] is False
    assert any("server /version did not report" in blocker for blocker in report["blockers"])


def test_remote_endpoint_url_is_redacted_in_the_artifact(tmp_path) -> None:
    report = validate_vllm_environment(
        model="test-model",
        base_url="http://vllm.internal.example:8000/v1",
        output_dir=tmp_path,
        timeout_seconds=0.5,
    )

    assert report["base_url"] == "redacted"
    written = (tmp_path / "vllm_validation.json").read_text(encoding="utf-8")
    assert "vllm.internal.example" not in written
    assert "vllm.internal.example" not in (tmp_path / "vllm_validation.md").read_text(encoding="utf-8")
