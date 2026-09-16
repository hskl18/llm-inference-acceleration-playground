"""Record the host and server state around a measurement phase.

The benchmark artifacts already carry the OS, Python, code commit, and the server version they
read from the endpoint. What they cannot see is whether the machine was busy with something
else, which on a shared laptop is the difference between a measurement and a guess. This writes
the load average, the CPU and memory shape, and the served model's digest so a reader can judge
the conditions instead of taking "the machine was idle" on trust.

    python capture_environment.py --phase before --output environment_before.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib import request


def _sysctl(name: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name], check=True, capture_output=True, text=True, timeout=5
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _get_json(url: str) -> object:
    try:
        with request.urlopen(request.Request(url, method="GET"), timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except OSError as exc:
        return {"error": str(exc)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--server-root", default="http://127.0.0.1:11434")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    load_1, load_5, load_15 = os.getloadavg()
    payload = {
        "phase": args.phase,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "cpu": _sysctl("machdep.cpu.brand_string"),
            "logical_cpus": _sysctl("hw.ncpu"),
            "performance_cpus": _sysctl("hw.perflevel0.logicalcpu"),
            "efficiency_cpus": _sysctl("hw.perflevel1.logicalcpu"),
            "memory_bytes": _sysctl("hw.memsize"),
            "platform": platform.platform(),
        },
        # The load average is the honest answer to "was anything else running?".
        "load_average": {"1m": load_1, "5m": load_5, "15m": load_15},
        "ollama_version": _get_json(f"{args.server_root}/api/version"),
        "ollama_loaded_models": _get_json(f"{args.server_root}/api/ps"),
        "ollama_available_models": _get_json(f"{args.server_root}/api/tags"),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "load_average_1m": load_1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
