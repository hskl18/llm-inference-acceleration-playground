# Release Process

This project is currently pre-release.
Use `0.x` versions until the CLI and result schemas are stable.

## Local Release Checklist

Run the full local release gate:

```bash
python scripts/release_check.py
```

The release gate checks authoritative package metadata, editable-install packaging, the console script version and help output, lint, tests, and the CLI smoke gate.

Manual equivalent:

1. Check release metadata:

   ```bash
   python scripts/release_check.py --metadata-only
   ```

2. Check packaging metadata and the console script:

   ```bash
python -m pip install -e .
   llm-accel --version
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

7. Run the deterministic mock matrix and confirm it remains workflow-only evidence:

   ```bash
   llm-accel bench matrix \
     --config configs/optimization_matrix_mock.yaml \
     --output-dir results/runs/release-mock-matrix
   llm-accel report ranking-audit --matrix-dir results/runs/release-mock-matrix
   ```

   The ranking audit must return nonzero because mock and closed-loop evidence cannot support a performance ranking.

8. Confirm `src/llm_accel/_version.py` contains the intended version and `pyproject.toml` derives package metadata from it.
9. Confirm `llm-accel --version` prints the intended version.
10. Confirm no mock measurement is described as model, backend, or hardware performance.
11. Confirm no tag or GitHub Release exists for an unmerged release pull request.
12. Review the generated changelog through the repository's release workflow rather than editing `CHANGELOG.md` manually.
13. Remove build directories, editable-install metadata, test caches, generated smoke runs, and temporary result bundles when they are safe to delete.
14. Tag only after validation passes and the release owner separately authorizes a release.

## Version Authority

`src/llm_accel/_version.py` is the single authoritative version source.
Setuptools derives installed package metadata from that module through `tool.setuptools.dynamic`.
Runtime metadata, CLI version output, manifests, and mock backend version reporting import the same value.

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

For a cross-profile ranking, also run the bundle gate:

```bash
llm-accel report ranking-audit --matrix-dir results/runs/hardware-matrix
```

The claim audit requires real endpoint evidence, an immutable model revision, a matching server-command file and fingerprint, accelerator and software versions, sufficient warmup and completed requests, raw request parity, p50/p95/p99 latency metrics, throughput, bounded errors, and GPU memory telemetry.
Passing one run is not enough for a comparison claim.
The ranking audit requires repeated compatible runs, common quality-suite fingerprints, quality deltas, open-loop dispatch evidence, and a non-saturated client.
It also requires exactly one compatible comparison stratum.
