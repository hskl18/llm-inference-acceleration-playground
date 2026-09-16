# Throughput Summary

## Run

- Model: `qwen2.5:1.5b-instruct`
- Backend: `ollama`
- Hardware label: `apple-m3-pro-18gb`
- Concurrency: `1`
- Input tokens: `378`
- Output tokens: `64`

## Throughput

- Output tokens/sec: `70.02028552494637`
- Requests/sec: `1.094066961327287`
- Measured elapsed seconds: `29.24866679200204`
- Completed requests: `32`
- Failed requests: `0`
- Timeout count: `0`

Raw per-request metrics remain in `raw_requests.jsonl` and `raw_requests.csv`.

## Warnings

- GPU memory telemetry unavailable: nvidia-smi not found.
- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
