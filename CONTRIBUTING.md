# Contributing

This project is early and should grow through small, reviewable changes.

Good first contribution areas:

- add a metric fixture
- improve report formatting
- add a benchmark config
- add method notes under `docs/methods/`
- improve mock endpoint behavior
- add tests for config validation or result schemas

Suggested issue labels:

- `backend`
- `benchmark`
- `documentation`
- `good first issue`
- `kv-cache`
- `metrics`
- `quantization`
- `speculative-decoding`
- `testing`

## Development Setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m ruff check .
python3 -m pytest
python3 scripts/smoke.py
```

## Quality Bar

Before proposing a substantial change, make sure:

- tests pass
- lint passes
- CLI behavior is documented
- result schema changes include fixture updates
- benchmark output does not fabricate hardware performance
- secrets are not written to result files

## Conduct and Security

See `CODE_OF_CONDUCT.md` for contributor behavior expectations and `SECURITY.md` for private reporting guidance. Keep public issues free of API keys, private prompts, and sensitive endpoint URLs.

## Release Hygiene

See `docs/release.md`. Do not publish benchmark claims from mock runs, incomplete manifests, or unvalidated run directories.

## Benchmark Integrity

Mock benchmark results are workflow examples only. Real hardware claims must include enough metadata to reproduce the run: model, backend, dtype, quantization mode, workload shape, concurrency, request count, timestamp, and hardware label where available.
