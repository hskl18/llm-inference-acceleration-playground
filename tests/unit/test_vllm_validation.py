import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_accel.serving.vllm_validation import validate_vllm_environment


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
    class Unauthorized(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Unauthorized)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        report = validate_vllm_environment(model="test-model", base_url=base_url, output_dir=tmp_path)
    finally:
        server.shutdown()

    assert report["checks"]["endpoint_health"]["status"] == "unauthorized"
    assert any(blocker.startswith("endpoint health check unauthorized") for blocker in report["blockers"])
