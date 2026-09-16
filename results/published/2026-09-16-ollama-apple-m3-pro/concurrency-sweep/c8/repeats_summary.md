# Repeated Benchmark Summary

- Model: `qwen2.5:1.5b-instruct`
- Backend: `ollama`
- Hardware label: `apple-m3-pro-18gb`
- Concurrency: `8`
- Output tokens: `64`
- Repetitions: `3`
- Failed requests across repetitions: `0`

| Metric | Mean | 95% CI | Sample stddev | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `goodput.attainment` | 0.031 | [0.031, 0.031] | 0.000 | 0.031 | 0.031 |
| `goodput.good_requests_per_second` | 0.030 | [0.022, 0.038] | 0.003 | 0.026 | 0.032 |
| `inter_token_latency_ms.p50` | 10.518 | [8.063, 12.974] | 0.988 | 9.377 | 11.102 |
| `inter_token_latency_ms.p95` | 15.305 | [12.584, 18.027] | 1.095 | 14.106 | 16.252 |
| `latency_ms.p50` | 8312.644 | [5516.430, 11108.858] | 1125.537 | 7278.765 | 9511.605 |
| `latency_ms.p95` | 9052.449 | [6658.499, 11446.399] | 963.617 | 8284.137 | 10133.620 |
| `throughput.output_tokens_per_second` | 61.095 | [44.274, 77.915] | 6.771 | 53.502 | 66.505 |
| `throughput.requests_per_second` | 0.955 | [0.692, 1.217] | 0.106 | 0.836 | 1.039 |
| `tpot_ms.p50` | 10.914 | [8.066, 13.763] | 1.147 | 9.601 | 11.719 |
| `ttft_ms.p50` | 7601.835 | [4910.486, 10293.183] | 1083.326 | 6666.484 | 8788.840 |
| `ttft_ms.p95` | 8367.286 | [5938.289, 10796.284] | 977.724 | 7675.918 | 9485.921 |

## Warnings

- Closed-loop scheduling is susceptible to coordinated omission because new requests depend on prior completions.
- GPU memory telemetry unavailable: nvidia-smi not found.

## Notes

- Each repetition is an independent run in its own directory under this one.
- Confidence intervals are two-sided 95% Student t intervals for the mean of the repetitions.
- Repetitions run back to back on one host, so they capture run-to-run noise, not day-to-day drift.
