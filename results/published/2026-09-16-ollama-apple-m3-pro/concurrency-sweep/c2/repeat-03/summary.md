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
- Workload fingerprint: `e4ac46f86d8e8308`
- Shared prefix tokens estimate: `4`
- Shared prefix fingerprint: `d8f0222550f8dcb5`
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
| Latency p50 | 2055.345 ms |
| Latency p95 | 3461.954 ms |
| Latency p99 | 4033.997 ms |
| TTFT p50 | 1344.828 ms |
| TTFT p95 | 2120.498 ms |
| TTFT p99 | 2527.473 ms |
| TPOT p50 | 10.493 ms |
| TPOT p95 | 20.849 ms |
| TPOT p99 | 26.492 ms |
| Inter-token latency samples | 2016 |
| Inter-token latency p50 | 10.200 ms |
| Inter-token latency p95 | 19.056 ms |
| Inter-token latency p99 | 43.213 ms |
| Queue delay p50 | 0.002 ms |
| Queue delay p95 | 0.018 ms |
| Queue delay p99 | 0.030 ms |
| End-to-end latency p50 | 2055.348 ms |
| End-to-end latency p95 | 3461.966 ms |
| End-to-end latency p99 | 4034.021 ms |
| Output tokens/sec | 56.832 |
| Requests/sec | 0.888 |
| Client CPU cores used | 0.009 |

## Goodput

- SLO: `tpot_ms <= 15 ms, ttft_ms <= 1000 ms`
- Requests meeting the SLO: `1`
- SLO attainment: `0.031`
- Good requests/sec: `0.028`
- Good output tokens/sec: `1.776`

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
