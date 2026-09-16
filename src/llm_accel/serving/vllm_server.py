"""Read backend identity and serving state from the vLLM server instead of the client host.

vLLM serves `GET /version` (`{"version": ...}`) and a Prometheus `GET /metrics` endpoint at the
server root, so a benchmark client on another machine can still record what actually served it.
"""

from __future__ import annotations

import json
import re
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request

from llm_accel.serving.openai_client import DEFAULT_API_KEY_ENV, bearer_auth_header


SPEC_DECODE_METRICS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
PREFIX_CACHE_METRICS = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
)
_SAMPLE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?P<labels>\{.*\})?\s+(?P<value>\S+)$")


def server_root(base_url: str) -> str:
    """The scheme and authority of an OpenAI-compatible base URL; /version and /metrics live there."""
    parsed = urllib_parse.urlsplit(base_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"base_url {base_url!r} must be an absolute http(s) URL")
    return urllib_parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def fetch_server_version(
    base_url: str,
    *,
    timeout_seconds: float = 5.0,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> str | None:
    """The version string reported by the serving process, or None when it cannot be read."""
    try:
        body = _get(f"{server_root(base_url)}/version", timeout_seconds, api_key_env)
        payload = json.loads(body)
    except (ValueError, OSError, urllib_error.URLError):
        return None
    version = payload.get("version") if isinstance(payload, dict) else None
    return version if isinstance(version, str) and version.strip() else None


def fetch_serving_state(
    base_url: str,
    *,
    timeout_seconds: float = 5.0,
    api_key_env: str = DEFAULT_API_KEY_ENV,
) -> dict[str, object]:
    """Speculative-decoding and prefix-cache counters scraped from the server's /metrics endpoint."""
    try:
        text = _get(f"{server_root(base_url)}/metrics", timeout_seconds, api_key_env)
    except (ValueError, OSError, urllib_error.URLError) as exc:
        return {"available": False, "error": str(exc), "counters": {}, "acceptance": None, "prefix_cache_hit_rate": None}
    counters = parse_prometheus_counters(text, SPEC_DECODE_METRICS + PREFIX_CACHE_METRICS)
    return {
        "available": True,
        "error": None,
        "counters": counters,
        "acceptance": spec_decode_acceptance(counters),
        "prefix_cache_hit_rate": _ratio(
            counters.get("vllm:prefix_cache_hits"),
            counters.get("vllm:prefix_cache_queries"),
        ),
    }


def parse_prometheus_counters(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """Sum every label set of the requested metric families ("_total" counter suffix included)."""
    wanted = {name: 0.0 for name in names}
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SAMPLE.match(stripped)
        if match is None:
            continue
        family = match.group("name").removesuffix("_total")
        if family not in wanted:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        wanted[family] += value
        seen.add(family)
    return {name: value for name, value in wanted.items() if name in seen}


def spec_decode_acceptance(counters: dict[str, float]) -> dict[str, float] | None:
    """Measured acceptance rate and accepted tokens per draft, or None without draft counters."""
    draft_tokens = counters.get("vllm:spec_decode_num_draft_tokens")
    accepted_tokens = counters.get("vllm:spec_decode_num_accepted_tokens")
    if draft_tokens is None or accepted_tokens is None or draft_tokens <= 0:
        return None
    drafts = counters.get("vllm:spec_decode_num_drafts")
    acceptance = {
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted_tokens,
        "acceptance_rate": accepted_tokens / draft_tokens,
    }
    if drafts is not None and drafts > 0:
        acceptance["num_drafts"] = drafts
        acceptance["draft_tokens_per_draft"] = draft_tokens / drafts
        acceptance["accepted_tokens_per_draft"] = accepted_tokens / drafts
    return acceptance


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _get(url: str, timeout_seconds: float, api_key_env: str) -> str:
    req = request.Request(url, method="GET", headers={"Accept": "*/*", **bearer_auth_header(api_key_env)})
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return response.read().decode("utf-8")
