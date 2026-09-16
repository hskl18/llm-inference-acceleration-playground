# Repeated Benchmark Summary

- Model: `qwen2.5:1.5b-instruct`
- Backend: `ollama`
- Hardware label: `apple-m3-pro-18gb`
- Concurrency: `4`
- Output tokens: `64`
- Repetitions: `3`
- Failed requests across repetitions: `0`

| Metric | Mean | 95% CI | Sample stddev | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `goodput.attainment` | 0.021 | [-0.024, 0.066] | 0.018 | 0.000 | 0.031 |
| `goodput.good_requests_per_second` | 0.022 | [-0.025, 0.069] | 0.019 | 0.000 | 0.034 |
| `inter_token_latency_ms.p50` | 10.571 | [7.203, 13.939] | 1.356 | 9.512 | 12.099 |
| `inter_token_latency_ms.p95` | 14.999 | [7.579, 22.420] | 2.987 | 12.124 | 18.087 |
| `latency_ms.p50` | 3996.815 | [2801.263, 5192.367] | 481.236 | 3715.043 | 4552.480 |
| `latency_ms.p95` | 4824.974 | [2095.380, 7554.569] | 1098.721 | 3835.380 | 6007.314 |
| `throughput.output_tokens_per_second` | 62.674 | [43.073, 82.275] | 7.890 | 53.739 | 68.683 |
| `throughput.requests_per_second` | 0.979 | [0.673, 1.286] | 0.123 | 0.840 | 1.073 |
| `tpot_ms.p50` | 10.926 | [6.858, 14.994] | 1.637 | 9.709 | 12.788 |
| `ttft_ms.p50` | 3306.032 | [2304.012, 4308.052] | 403.335 | 3050.057 | 3770.971 |
| `ttft_ms.p95` | 4037.188 | [1619.176, 6455.200] | 973.302 | 3181.412 | 5095.990 |

## Warnings

- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
- GPU memory telemetry unavailable: nvidia-smi not found.

## Notes

- Each repetition is an independent run in its own directory under this one.
- Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.
- Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.
