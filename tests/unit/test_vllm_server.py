import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_accel.serving.versions import resolve_backend_version
from llm_accel.serving.vllm_server import (
    fetch_server_version,
    fetch_serving_state,
    parse_prometheus_counters,
    server_root,
    spec_decode_acceptance,
)
from llm_accel.speculative_decoding.analytic import read_vllm_acceptance


METRICS_BODY = """# HELP vllm:spec_decode_num_drafts_total Number of drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{model_name="demo"} 100.0
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{model_name="demo"} 400.0
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="demo"} 300.0
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{model_name="demo"} 50.0
vllm:prefix_cache_hits_total{model_name="demo"} 20.0
vllm:num_requests_running{model_name="demo"} 3.0
"""


class _FakeVllm(BaseHTTPRequestHandler):
    version = "0.29.0"
    serve_metrics = True

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/version":
            self._send(200, json.dumps({"version": self.version}).encode("utf-8"), "application/json")
        elif self.path == "/metrics" and self.serve_metrics:
            self._send(200, METRICS_BODY.encode("utf-8"), "text/plain; version=0.0.4")
        else:
            self._send(404, b"{}", "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture()
def fake_vllm_base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVllm)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


def test_server_root_drops_the_openai_path_prefix() -> None:
    assert server_root("http://example.test:8000/v1") == "http://example.test:8000"
    assert server_root("https://example.test/v1/") == "https://example.test"
    with pytest.raises(ValueError):
        server_root("not-a-url")


def test_fetch_server_version_reads_the_serving_process(fake_vllm_base_url: str) -> None:
    assert fetch_server_version(fake_vllm_base_url) == "0.29.0"


def test_backend_version_prefers_the_server_over_the_client_package(fake_vllm_base_url: str) -> None:
    version, source = resolve_backend_version("vllm", fake_vllm_base_url)

    assert version == "0.29.0"
    assert source == "server_version_endpoint"


def test_backend_version_falls_back_when_the_server_cannot_be_reached() -> None:
    version, source = resolve_backend_version("vllm", "http://127.0.0.1:1/v1", timeout_seconds=0.5)

    assert source in {"client_package", "unavailable"}
    assert (version is None) == (source == "unavailable")


def test_mock_endpoint_is_never_probed() -> None:
    version, source = resolve_backend_version("openai-compatible", "mock://local")

    assert source == "mock"
    assert version.startswith("llm-accel-mock/")


def test_serving_state_reports_spec_decode_and_prefix_cache_counters(fake_vllm_base_url: str) -> None:
    state = fetch_serving_state(fake_vllm_base_url)

    assert state["available"] is True
    assert state["counters"]["vllm:spec_decode_num_draft_tokens"] == 400.0
    assert state["acceptance"]["acceptance_rate"] == 0.75
    assert state["acceptance"]["draft_tokens_per_draft"] == 4.0
    assert state["prefix_cache_hit_rate"] == 0.4


def test_read_vllm_acceptance_exposes_the_measured_rate(fake_vllm_base_url: str) -> None:
    measured = read_vllm_acceptance(fake_vllm_base_url)

    assert measured["metrics_available"] is True
    assert measured["acceptance_rate"] == 0.75
    assert measured["observed_lookahead"] == 4.0


def test_read_vllm_acceptance_reports_an_unreachable_endpoint() -> None:
    measured = read_vllm_acceptance("http://127.0.0.1:1/v1", timeout_seconds=0.5)

    assert measured["metrics_available"] is False
    assert measured["acceptance_rate"] is None
    assert measured["error"]


def test_prometheus_parser_ignores_help_lines_and_unrelated_metrics() -> None:
    counters = parse_prometheus_counters(METRICS_BODY, ("vllm:prefix_cache_hits", "vllm:missing"))

    assert counters == {"vllm:prefix_cache_hits": 20.0}


def test_acceptance_needs_draft_token_counters() -> None:
    assert spec_decode_acceptance({"vllm:spec_decode_num_accepted_tokens": 1.0}) is None
    assert spec_decode_acceptance({
        "vllm:spec_decode_num_draft_tokens": 0.0,
        "vllm:spec_decode_num_accepted_tokens": 0.0,
    }) is None
