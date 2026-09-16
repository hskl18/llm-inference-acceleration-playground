# Repeated Benchmark Summary

- Model: `qwen2.5:1.5b-instruct`
- Backend: `ollama`
- Hardware label: `apple-m3-pro-18gb`
- Concurrency: `2`
- Output tokens: `64`
- Repetitions: `3`
- Failed requests across repetitions: `0`

| Metric | Mean | 95% CI | Sample stddev | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `goodput.attainment` | 0.031 | [0.031, 0.031] | 0.000 | 0.031 | 0.031 |
| `goodput.good_requests_per_second` | 0.031 | [0.023, 0.038] | 0.003 | 0.028 | 0.034 |
| `inter_token_latency_ms.p50` | 10.194 | [10.007, 10.381] | 0.075 | 10.116 | 10.266 |
| `inter_token_latency_ms.p95` | 15.148 | [6.352, 23.944] | 3.540 | 12.155 | 19.056 |
| `latency_ms.p50` | 1942.159 | [1648.857, 2235.461] | 118.060 | 1819.764 | 2055.345 |
| `latency_ms.p95` | 2578.932 | [632.530, 4525.333] | 783.469 | 1967.049 | 3461.954 |
| `throughput.output_tokens_per_second` | 62.921 | [47.777, 78.066] | 6.096 | 56.832 | 69.024 |
| `throughput.requests_per_second` | 0.983 | [0.747, 1.220] | 0.095 | 0.888 | 1.079 |
| `tpot_ms.p50` | 10.411 | [9.948, 10.875] | 0.187 | 10.198 | 10.544 |
| `ttft_ms.p50` | 1265.771 | [1059.071, 1472.471] | 83.201 | 1178.968 | 1344.828 |
| `ttft_ms.p95` | 1650.379 | [584.165, 2716.593] | 429.174 | 1279.556 | 2120.498 |

## Warnings

- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
- GPU memory telemetry unavailable: nvidia-smi not found.

## Notes

- Each repetition is an independent run in its own directory under this one.
- Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.
- Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.
