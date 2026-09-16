# Throughput Summary

## Run

- Model: `qwen2.5:1.5b-instruct`
- Backend: `ollama`
- Hardware label: `apple-m3-pro-18gb`
- Concurrency: `2`
- Input tokens: `378`
- Output tokens: `64`

## Throughput

- Output tokens/sec: `69.02436117163008`
- Requests/sec: `1.07850564330672`
- Measured elapsed seconds: `29.670683875039686`
- Completed requests: `32`
- Failed requests: `0`
- Timeout count: `0`

Raw per-request metrics remain in `raw_requests.jsonl` and `raw_requests.csv`.

## Warnings

- GPU memory telemetry unavailable: nvidia-smi not found.
- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
