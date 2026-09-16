"""Generate the prompt sets this study measured.

Ollama keeps recently served prompts in its runner cache, so re-sending a prompt measures a
cache hit rather than the model. Every run in this study therefore needs its own prompt set
that the server has never seen, which is what the `--run-id` namespace below provides.

`--run-id` must be unique across executions, not only across the runs of one execution.
Re-running a study with the run ids it used last time hands the server prompts it still has
cached, which is exactly the contamination the namespace exists to prevent. Include an
execution tag, as the published runs did with their `-x1` suffix.

The two arms are matched on length and vocabulary and differ only in prefix sharing:

- `shared`: one common opening block, then the same ten handbook sentences in a fixed order,
  then a per-prompt question. Every prompt in a run shares a long exact prefix.
- `unique`: the same ten sentences in a per-prompt shuffled order behind a per-prompt marker,
  so two prompts in a run diverge within a handful of tokens.

Each run's `run_metadata.json` records a `workload_fingerprint`, so regenerating with the same
arguments and comparing fingerprints proves which prompt set a published run measured.

    python build_prompts.py --arm unique --run-id sweep-c1-repeat-01 --output prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SENTENCES = (
    "The section explains that a serving engine splits every request into a prefill phase, which processes the whole prompt in one"
    " forward pass, and a decode phase, which produces one token per forward pass.",
    "It notes that prefill is compute bound while decode is memory bandwidth bound, because decode must re-read the entire key value"
    " cache for every token it emits.",
    "It describes paged attention, which stores the key value cache in fixed size blocks so the engine can allocate memory without"
    " reserving a contiguous region for the longest possible sequence.",
    "It describes continuous batching, which admits a new request into the running batch as soon as another request finishes instead"
    " of waiting for the whole batch to drain.",
    "It describes prefix caching, which keeps the key value cache blocks of an already processed prompt prefix so a later request"
    " that shares that prefix can skip re-computing it.",
    "It warns that a benchmark whose synthetic prompts repeat will silently measure prefix cache hits rather than model throughput.",
    "It warns that a load generator which saturates its own CPU will report client queueing as server latency.",
    "It warns that closed loop benchmarks suffer from coordinated omission, because a slow response delays the arrival of the next request and hides the tail.",
    "It reminds the reader that a fixed output length is required before token throughput can be compared across two configurations.",
    "It reminds the reader that percentiles over requests describe one run and say nothing about how much the result moves when the run is repeated.",
)
HEAD = "You are reviewing an internal engineering handbook section on large language model inference serving."
TAIL = "Answer the following question about that section in a detailed paragraph of at least eighty words. Question: "
QUESTIONS = (
    "Why is the decode phase bound by memory bandwidth rather than by compute?",
    "What problem does paged attention solve that a contiguous allocation does not?",
    "How does continuous batching change the throughput of a busy server?",
    "Why can a repeating synthetic workload overstate measured throughput?",
    "What does coordinated omission hide in a closed loop benchmark?",
    "Why does a saturated load generator misattribute latency to the server?",
    "What has to be true for a later request to reuse a cached prompt prefix?",
    "Why does prefill behave differently from decode on the same hardware?",
    "What does a key value cache store and why does its size matter?",
    "Why does one forward pass per token limit single request latency?",
    "How does block based cache allocation reduce memory fragmentation?",
    "What evidence would show that a benchmark measured cache hits?",
    "Why does admitting requests mid batch help tail latency?",
    "What distinguishes a compute bound phase from a bandwidth bound phase?",
    "Why is per request latency a poor proxy for server throughput?",
    "What should a report state when output lengths differ across requests?",
    "How does a longer prompt change the cost of the prefill phase?",
    "Why does a shared prefix help only when it is an exact prefix?",
    "What happens to queued requests when a server admits one at a time?",
    "Why does time to first token grow when requests wait in a queue?",
    "How would you tell server queueing apart from client queueing?",
    "Why does an early stop token make token throughput incomparable?",
    "What does a confidence interval add to a single benchmark number?",
    "Why does one averaged token latency hide a mid response stall?",
    "How does the key value cache size scale with sequence length?",
    "Why is a benchmark on one machine not a claim about another?",
    "What makes a quantized model comparison hard to interpret fairly?",
    "Why does the engine need to re-read the cache for every token?",
    "How does batch admission interact with memory pressure on a server?",
    "Why should a benchmark record the exact serving command it measured?",
    "What does goodput measure that average latency does not?",
    "Why does a warmup request change what a cache holds?",
)
# Every prompt opens with three copies of one marker word. A word repeated three times keeps the
# two arms the same length while giving the unique arm a distinct first token.
MARKERS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu ember fjord "
    "gamut hazel ionic jasper"
).split()


def build_prompts(arm: str, run_id: str, count: int) -> list[str]:
    if arm not in {"shared", "unique"}:
        raise ValueError("arm must be 'shared' or 'unique'")
    if not 1 <= count <= len(QUESTIONS):
        raise ValueError(f"count must be between 1 and {len(QUESTIONS)}")
    prompts = []
    for index in range(count):
        question = QUESTIONS[index]
        body = list(SENTENCES)
        if arm == "shared":
            opening = [f"Handbook revision note {run_id} shared shared shared.", HEAD]
        else:
            marker = MARKERS[index]
            random.Random(f"{run_id}:{index}").shuffle(body)
            opening = [f"Handbook revision note {run_id} {marker} {marker} {marker}."]
            # HEAD moves to the end so the shuffled sentences start the divergence immediately.
            body.append(HEAD)
        prompts.append(" ".join([*opening, *body, TAIL + question]))
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["shared", "unique"], required=True)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Namespace that makes this run's prompts unseen; must be unique across executions too",
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    prompts = build_prompts(args.arm, args.run_id, args.count)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(path), "arm": args.arm, "run_id": args.run_id, "count": len(prompts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
