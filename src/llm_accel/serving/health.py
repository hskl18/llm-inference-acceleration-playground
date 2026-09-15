from __future__ import annotations

import json
import time
from urllib import error as urllib_error
from urllib import request


def check_endpoint_health(base_url: str, timeout_seconds: float = 5.0) -> dict[str, object]:
    """Probe GET {base_url}/models.

    status is "healthy" only for a 2xx response with a JSON body, "unauthorized" for 401 or 403,
    "unhealthy" for other HTTP errors or a non-JSON body, and "unreachable" when no HTTP response arrives.
    """
    if base_url.startswith("mock://"):
        return {
            "base_url": base_url,
            "healthy": True,
            "status": "healthy",
            "status_code": None,
            "latency_ms": 0.0,
            "error": None,
        }

    started = time.perf_counter()
    url = f"{base_url.rstrip('/')}/models"
    req = request.Request(url, method="GET", headers={"Accept": "application/json"})
    status_code: int | None = None
    error: str | None = None
    try:
        # urlopen raises HTTPError for every non-2xx status.
        with request.urlopen(req, timeout=timeout_seconds) as response:
            status_code = response.status
            json.loads(response.read().decode("utf-8"))
        status = "healthy"
    except urllib_error.HTTPError as exc:
        status_code = exc.code
        status = "unauthorized" if exc.code in {401, 403} else "unhealthy"
        error = f"HTTP {exc.code} from /models"
    except ValueError as exc:
        status = "unhealthy"
        error = f"/models did not return JSON: {exc}"
    except (urllib_error.URLError, OSError) as exc:
        status = "unreachable"
        error = str(exc)
    return {
        "base_url": base_url if base_url.startswith(("http://localhost", "http://127.0.0.1")) else "redacted",
        "healthy": status == "healthy",
        "status": status,
        "status_code": status_code,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "error": error,
    }
