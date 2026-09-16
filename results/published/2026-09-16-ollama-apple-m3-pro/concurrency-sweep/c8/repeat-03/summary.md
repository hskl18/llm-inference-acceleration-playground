# Benchmark Summary

## Run Metadata

- Model: `qwen2.5:1.5b-instruct`
- Model revision: `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`
- Backend: `ollama`
- Backend version: `0.34.1`
- API kind: `chat`
- Concurrency: `8`
- Input tokens: `378`
- Output tokens: `64`
- Workload mode: `fixed_prompts`
- Prompt count: `32`
- Workload fingerprint: `f56738a6c0481fef`
- Shared prefix tokens estimate: `4`
- Shared prefix fingerprint: `b4218d20161d303f`
- Measured requests: `32`
- Warmup requests: `0`
- Request schedule: `closed-loop`
- Request rate: `n/a` requests/sec
- Client processes: `1`
- Client workers: `8`
- Queue delay warning threshold: `10.0` ms
- Hardware label: `apple-m3-pro-18gb`
- Optimization profile: `ollama-default`
- Server command SHA-256: `unavailable`
- GPU name: `unavailable`
- GPU driver: `unavailable`
- CUDA: `unavailable`
- NVIDIA driver CUDA API: `unavailable`
- PyTorch: `unavailable`
- Python: `3.11.15`
- OS: `macOS-27.0-arm64-arm-64bit`
- Git commit: `140f2543bef53d40d0c3ffb3057db1ea01988249`

## Metrics

| Metric | Value |
| --- | ---: |
| Completed requests | 32 |
| Failed requests | 0 |
| Timeout count | 0 |
| Latency p50 | 7278.765 ms |
| Latency p95 | 8284.137 ms |
| Latency p99 | 8470.318 ms |
| TTFT p50 | 6666.484 ms |
| TTFT p95 | 7675.918 ms |
| TTFT p99 | 7862.024 ms |
| TPOT p50 | 9.601 ms |
| TPOT p95 | 14.108 ms |
| TPOT p99 | 15.069 ms |
| Inter-token latency samples | 2016 |
| Inter-token latency p50 | 9.377 ms |
| Inter-token latency p95 | 15.559 ms |
| Inter-token latency p99 | 17.191 ms |
| Queue delay p50 | 0.003 ms |
| Queue delay p95 | 0.004 ms |
| Queue delay p99 | 0.007 ms |
| End-to-end latency p50 | 7278.768 ms |
| End-to-end latency p95 | 8284.142 ms |
| End-to-end latency p99 | 8470.321 ms |
| Output tokens/sec | 66.505 |
| Requests/sec | 1.039 |
| Client CPU cores used | 0.009 |

## Goodput

- SLO: `tpot_ms <= 15 ms, ttft_ms <= 1000 ms`
- Requests meeting the SLO: `1`
- SLO attainment: `0.031`
- Good requests/sec: `0.032`
- Good output tokens/sec: `2.078`

## Memory

- GPU memory telemetry available: `False`
- Delta used memory: `None` MiB

## Warnings

- GPU memory telemetry unavailable: nvidia-smi not found.
- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.

## Artifacts

- Raw request records: `raw_requests.jsonl`
- Raw request CSV: `raw_requests.csv`
- Machine-readable summary: `summary.json`
- Latency plot: `plots/latency.svg`

## Notes

- This report separates measured request data from later analysis.
- Mock backend results are examples for workflow validation, not hardware performance claims.
