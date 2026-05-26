# Quantization

Quantization results should only be reported when the backend and hardware actually support the selected mode.

Reports should separate:

- memory savings
- TTFT changes
- TPOT and throughput changes
- p95 latency changes
- fixed-prompt sanity checks

Unsupported modes should be reported explicitly rather than hidden.

The current implementation includes:

```bash
llm-accel quantization compare --modes none,int8,int4
```

The mock backend validates the workflow and report shape only. Real quantization claims require a backend that actually runs each mode.

Comparison reports include a fixed-prompt quality sanity check for each mode. The check is intentionally lightweight: it verifies that outputs are non-empty and records output length and errors. It does not replace a task-specific evaluation.

Each requested mode is labeled with a support status:

- `supported`: listed in the backend capability matrix and benchmarked.
- `unsupported`: not listed for the backend; reported but not benchmarked.
- `unknown`: backend capabilities are not known locally; benchmark output is endpoint-defined.

Unsupported modes do not produce throughput or latency claims.
