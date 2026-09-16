import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_accel.quantization.comparison import compare_quantization_modes
from llm_accel.quantization.sanity import measure_prompt_perplexity


def test_quantization_comparison_marks_unsupported_modes(tmp_path) -> None:
    report = compare_quantization_modes(
        model="mock-model",
        mode_endpoints={"none": "mock://q-none", "fp8": "mock://q-fp8"},
        output_dir=tmp_path,
        request_count=1,
        backend="mock",
    )

    supported, unsupported = report["runs"]

    assert supported["quantization"] == "none"
    assert supported["support_status"] == "supported"
    assert supported["measured"] is True
    assert unsupported["quantization"] == "fp8"
    assert unsupported["support_status"] == "unsupported"
    assert unsupported["measured"] is False
    assert unsupported["summary_path"] is None
    assert any("not listed as supported" in warning for warning in report["warnings"])
    assert not (tmp_path / "fp8" / "summary.json").exists()


def test_each_mode_needs_its_own_endpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="share one endpoint"):
        compare_quantization_modes(
            model="mock-model",
            mode_endpoints={"none": "mock://same", "fp8": "mock://same/"},
            output_dir=tmp_path,
            request_count=1,
            backend="mock",
        )


def test_modes_are_scored_against_the_baseline_mode(tmp_path) -> None:
    report = compare_quantization_modes(
        model="mock-model",
        mode_endpoints={"none": "mock://q-none", "fp8": "mock://q-fp8"},
        output_dir=tmp_path,
        request_count=1,
        backend="openai-compatible",
    )

    baseline, treatment = report["runs"]

    assert report["baseline_mode"] == "none"
    assert baseline["endpoint_sha256"] != treatment["endpoint_sha256"]
    assert baseline["quality_vs_baseline"]["is_baseline"] is True
    # The deterministic mock client answers identically, so the treatment must match exactly.
    assert treatment["quality_vs_baseline"]["exact_match_rate"] == 1.0
    assert treatment["quality_vs_baseline"]["prompt_logprobs_supported"] is False
    assert treatment["quality_vs_baseline"]["perplexity_delta_from_baseline"] is None
    assert any("does not support echoed prompt logprobs" in warning for warning in report["warnings"])
    assert treatment["relative_to_baseline_mode"] is not None


def test_no_modes_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one quantization mode"):
        compare_quantization_modes(model="mock-model", mode_endpoints={}, output_dir=tmp_path)


class _LogprobServer(BaseHTTPRequestHandler):
    token_logprobs = [None, -0.5, -1.5]

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps(
            {"choices": [{"logprobs": {"token_logprobs": self.token_logprobs}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_perplexity_uses_echoed_prompt_logprobs() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _LogprobServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = measure_prompt_perplexity(
            base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="demo",
            text="fixed scoring text",
        )
    finally:
        server.shutdown()

    assert result["supported"] is True
    assert result["token_count"] == 2
    # exp(-mean(-0.5, -1.5)) == exp(1.0)
    assert round(result["perplexity"], 6) == round(2.718281828459045, 6)


def test_perplexity_reports_unsupported_backends() -> None:
    result = measure_prompt_perplexity(
        base_url="http://127.0.0.1:1/v1",
        model="demo",
        timeout_seconds=0.5,
    )

    assert result["supported"] is False
    assert result["perplexity"] is None
    assert result["error"]
