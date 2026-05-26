from __future__ import annotations

import json
import time
from urllib import request


def check_endpoint_health(base_url: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    if base_url.startswith("mock://"):
        return {
            "base_url": base_url,
            "healthy": True,
            "status_code": None,
            "latency_ms": 0.0,
            "error": None,
        }

    started = time.perf_counter()
    url = f"{base_url.rstrip('/')}/models"
    req = request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status_code = response.status
        json.loads(body) if body else None
        return {
            "base_url": base_url if base_url.startswith(("http://localhost", "http://127.0.0.1")) else "redacted",
            "healthy": 200 <= status_code < 500,
            "status_code": status_code,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": None,
        }
    except Exception as exc:
        return {
            "base_url": base_url if base_url.startswith(("http://localhost", "http://127.0.0.1")) else "redacted",
            "healthy": False,
            "status_code": None,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "error": str(exc),
        }
