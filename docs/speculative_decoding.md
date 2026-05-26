# Speculative Decoding

The current module is a toy accounting implementation. It exists to make acceptance rate, draft calls, target calls, rejection cost, and estimated speedup concrete.

It is not a production speculative decoding runtime.

Prompt files can be plain text lines or JSONL records containing a `prompt` field.

`llm-accel speculative run` writes:

- `speculative_summary.json`
- `speculative_summary.md`
- `acceptance_curve.json`
- `baseline_comparison.json`
- `baseline_comparison.md`

The acceptance curve varies the toy acceptance setting to show how estimated speedup changes as draft quality changes.

The baseline comparison contrasts target-only decoding steps with the toy speculative accounting. It is an explanatory model, not a measured serving benchmark.
