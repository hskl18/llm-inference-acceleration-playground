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
| `goodput.attainment` | 0.969 | [0.891, 1.046] | 0.031 | 0.938 | 1.000 |
| `goodput.good_requests_per_second` | 0.994 | [0.919, 1.069] | 0.030 | 0.963 | 1.024 |
| `inter_token_latency_ms.p50` | 10.034 | [9.753, 10.315] | 0.113 | 9.905 | 10.114 |
| `inter_token_latency_ms.p95` | 14.164 | [11.509, 16.820] | 1.069 | 13.053 | 15.185 |
| `latency_ms.p50` | 941.728 | [876.405, 1007.052] | 26.294 | 926.270 | 972.089 |
| `latency_ms.p95` | 1149.960 | [987.994, 1311.926] | 65.195 | 1078.598 | 1206.400 |
| `throughput.output_tokens_per_second` | 65.714 | [60.907, 70.520] | 1.935 | 63.757 | 67.626 |
| `throughput.requests_per_second` | 1.027 | [0.952, 1.102] | 0.030 | 0.996 | 1.057 |
| `tpot_ms.p50` | 10.246 | [10.156, 10.336] | 0.036 | 10.204 | 10.267 |
| `ttft_ms.p50` | 286.612 | [228.234, 344.990] | 23.499 | 271.724 | 313.701 |
| `ttft_ms.p95` | 341.213 | [277.145, 405.280] | 25.789 | 318.370 | 369.178 |

## Warnings

- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
- GPU memory telemetry unavailable: nvidia-smi not found.

## Notes

- Each repetition is an independent run in its own directory under this one.
- Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.
- Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.
