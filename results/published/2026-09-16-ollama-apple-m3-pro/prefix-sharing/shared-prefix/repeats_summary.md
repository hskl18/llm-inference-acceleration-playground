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
| `goodput.attainment` | 1.000 | [1.000, 1.000] | 0.000 | 1.000 | 1.000 |
| `goodput.good_requests_per_second` | 1.359 | [0.959, 1.759] | 0.161 | 1.211 | 1.531 |
| `inter_token_latency_ms.p50` | 10.376 | [7.473, 13.280] | 1.169 | 9.314 | 11.628 |
| `inter_token_latency_ms.p95` | 13.548 | [6.920, 20.176] | 2.668 | 10.468 | 15.151 |
| `latency_ms.p50` | 726.055 | [520.902, 931.208] | 82.578 | 643.544 | 808.701 |
| `latency_ms.p95` | 855.027 | [466.177, 1243.878] | 156.521 | 678.419 | 976.587 |
| `throughput.output_tokens_per_second` | 86.988 | [61.407, 112.569] | 10.297 | 77.529 | 97.956 |
| `throughput.requests_per_second` | 1.359 | [0.959, 1.759] | 0.161 | 1.211 | 1.531 |
| `tpot_ms.p50` | 10.728 | [7.600, 13.856] | 1.259 | 9.501 | 12.017 |
| `ttft_ms.p50` | 47.714 | [39.145, 56.284] | 3.450 | 44.965 | 51.585 |
| `ttft_ms.p95` | 58.296 | [30.269, 86.322] | 11.281 | 48.213 | 70.480 |

## Warnings

- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
- GPU memory telemetry unavailable: nvidia-smi not found.

## Notes

- Each repetition is an independent run in its own directory under this one.
- Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.
- Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.
