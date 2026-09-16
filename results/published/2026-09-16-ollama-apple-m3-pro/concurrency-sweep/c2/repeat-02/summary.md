# Benchmark Summary

## Run Metadata

- Model: `qwen2.5:1.5b-instruct`
- Model revision: `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`
- Backend: `ollama`
- Backend version: `0.34.1`
- API kind: `chat`
- Concurrency: `2`
- Input tokens: `378`
- Output tokens: `64`
- Workload mode: `fixed_prompts`
- Prompt count: `32`
- Workload fingerprint: `3e7c88ffaca2fa12`
- Shared prefix tokens estimate: `4`
- Shared prefix fingerprint: `c222af0845eca172`
- Measured requests: `32`
- Warmup requests: `0`
- Request schedule: `closed-loop`
- Request rate: `n/a` requests/sec
- Client processes: `1`
- Client workers: `2`
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
| Latency p50 | 1951.369 ms |
| Latency p95 | 2307.791 ms |
| Latency p99 | 2505.427 ms |
| TTFT p50 | 1273.517 ms |
| TTFT p95 | 1551.082 ms |
| TTFT p99 | 1674.797 ms |
| TPOT p50 | 10.544 ms |
| TPOT p95 | 13.412 ms |
| TPOT p99 | 16.264 ms |
| Inter-token latency samples | 2016 |
| Inter-token latency p50 | 10.266 ms |
| Inter-token latency p95 | 14.232 ms |
| Inter-token latency p99 | 22.261 ms |
| Queue delay p50 | 0.002 ms |
| Queue delay p95 | 0.006 ms |
| Queue delay p99 | 0.008 ms |
| End-to-end latency p50 | 1951.370 ms |
| End-to-end latency p95 | 2307.796 ms |
| End-to-end latency p99 | 2505.428 ms |
| Output tokens/sec | 62.907 |
| Requests/sec | 0.983 |
| Client CPU cores used | 0.009 |

## Goodput

- SLO: `tpot_ms <= 15 ms, ttft_ms <= 1000 ms`
- Requests meeting the SLO: `1`
- SLO attainment: `0.031`
- Good requests/sec: `0.031`
- Good output tokens/sec: `1.966`

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
