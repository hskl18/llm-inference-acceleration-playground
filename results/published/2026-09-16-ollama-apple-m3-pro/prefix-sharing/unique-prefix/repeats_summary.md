# Repeated Benchmark Summary

- Model: `qwen2.5:1.5b-instruct`
- Backend: `ollama`
- Hardware label: `apple-m3-pro-18gb`
- Concurrency: `1`
- Output tokens: `64`
- Repetitions: `3`
- Failed requests across repetitions: `0`

| Metric | Mean | 95% CI | Sample stddev | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `goodput.attainment` | 0.917 | [0.623, 1.211] | 0.118 | 0.781 | 1.000 |
| `goodput.good_requests_per_second` | 0.887 | [0.320, 1.454] | 0.228 | 0.642 | 1.094 |
| `inter_token_latency_ms.p50` | 10.353 | [7.864, 12.841] | 1.002 | 9.309 | 11.306 |
| `inter_token_latency_ms.p95` | 15.122 | [4.356, 25.888] | 4.333 | 10.383 | 18.883 |
| `latency_ms.p50` | 1054.644 | [714.574, 1394.714] | 136.886 | 913.107 | 1186.348 |
| `latency_ms.p95` | 1289.124 | [344.932, 2233.315] | 380.058 | 933.930 | 1689.929 |
| `throughput.output_tokens_per_second` | 61.224 | [39.592, 82.856] | 8.707 | 52.608 | 70.020 |
| `throughput.requests_per_second` | 0.957 | [0.619, 1.295] | 0.136 | 0.822 | 1.094 |
| `tpot_ms.p50` | 10.808 | [7.607, 14.010] | 1.289 | 9.522 | 12.100 |
| `ttft_ms.p50` | 316.605 | [267.908, 365.302] | 19.601 | 300.556 | 338.451 |
| `ttft_ms.p95` | 483.502 | [-161.299, 1128.302] | 259.546 | 333.589 | 783.200 |

## Warnings

- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
- GPU memory telemetry unavailable: nvidia-smi not found.

## Notes

- Each repetition is an independent run in its own directory under this one.
- Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.
- Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.
