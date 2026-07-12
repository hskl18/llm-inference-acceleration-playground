# Release Process

This project is currently pre-release. Use `0.x` versions until the CLI and result schemas are stable.

## Local Release Checklist

Run the full local release gate:

```bash
python scripts/release_check.py
```

The release gate checks package metadata, changelog coverage, editable-install packaging, the `llm-accel --help` console script, lint, tests, and the CLI smoke gate.

Manual equivalent:

1. Check release metadata:

   ```bash
   python scripts/release_check.py --metadata-only
   ```

2. Check packaging metadata and the console script:

   ```bash
   python -m pip install -e . --dry-run
   llm-accel --help
   ```

3. Run the test suite:

   ```bash
   python -m pytest
   ```

4. Run lint:

   ```bash
   python -m ruff check .
   ```

5. Run the CLI smoke gate:

   ```bash
   python scripts/smoke.py
   ```

6. Check generated artifacts:

   ```bash
   llm-accel report validate --run-dir results/runs/smoke-local
   ```

7. Update `CHANGELOG.md`.
8. Confirm `pyproject.toml`, `src/llm_accel/__init__.py`, and `CHANGELOG.md` refer to the same release version.
9. Tag only after validation passes.

## Versioning Notes

- Patch releases should not break result schemas.
- Minor releases may add metrics, commands, or optional artifacts.
- Breaking schema changes must update `schema_version` and include migration notes.

## Benchmark Claims

Do not publish hardware performance claims unless the run directory includes:

- manifest
- resolved config
- raw request records
- summary
- backend and model metadata
- backend version when available
- Python, OS, git commit, and hardware label metadata
- warning list for unsupported or missing measurements
- hardware/GPU telemetry when available
- validation output

For a hardware result, run the stricter gate:

```bash
llm-accel report claim-audit --run-dir results/runs/hardware-run
```

The claim audit requires real endpoint evidence, an immutable model revision, a matching server-command file and fingerprint, accelerator and software versions, sufficient warmup and completed requests, raw request parity, p50/p95/p99 latency metrics, throughput, bounded errors, and GPU memory telemetry.
Passing one run is not enough for a comparison claim.
Use repeated compatible runs and publish quality evidence separately.
