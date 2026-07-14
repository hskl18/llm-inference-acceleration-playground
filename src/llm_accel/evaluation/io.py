from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from llm_accel.metrics.io import write_lines_atomic


def write_mapping_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    write_lines_atomic(
        path,
        (json.dumps(dict(record), sort_keys=True) + "\n" for record in records),
    )
