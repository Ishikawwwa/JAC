# Performance model: H100 decisions and a remote LLM

These are **projections**, not JAC measurements. There is no universal average
remote LLM latency: provider, load, prompt length, output length and reasoning all
matter. The examples below use an assumed mean completion latency for a workload,
independent of any particular remote model. Completion means the full returned
response, not just the first streamed token.

## Reference deployment

- A warm JAC process with a persistent cache and one retrieved candidate.
- JevK5 4B, bf16, on an H100 using the upstream CUDA graph runtime.
- Two independent decision checks, plus retrieval and application overhead.
- A remote generation API; assume 5 seconds mean completion latency.
- Assume 100 ms mean total retrieval/decision overhead on every modeled request.

[JevK5's author reports](https://github.com/allebee/jevk5#results-on-jevbenchs-public-items)
in-process, batch-one H100 inference of 13.2 ms p50 / 14.6 ms p95 for short
questions. For hard questions with 1–4k-token documents, the reported values are
30 ms p50 / 162 ms p95. Those measurements describe upstream inference, not JAC
or a network service. Multiple questions and candidates require multiple passes.
The 100 ms end-to-end assumption must be measured on the actual deployment;
long prompts, cold starts, queueing and cross-region calls can exceed it.

JAC provides `JevK5CudaDecider` for this upstream runtime, plus a separate
`JevK5Decider` for GGUF/llama.cpp. CUDA adapter protocol tests use injected runtimes;
an H100 deployment and its end-to-end latency have not been benchmarked by JAC.
See [setup](JEVK5.md).

## Cache-first calculation

Let `G` be mean remote generation latency, `D` total lookup/validation overhead,
and `h` the fraction of requests that yield correct, eligible cache hits.
Assume comparable generation times for hits and misses and that every modeled
request pays `D`:

```text
Mean latency with JAC = D + (1 - h) × G
Mean latency saved   = h × G - D
Generation calls avoided = h × request count
Break-even hit rate = D / G
```

With `G = 5 s` and `D = 0.10 s`:

| Valid hit rate | Mean response latency | Mean latency reduction | Generation calls avoided |
| --- | ---: | ---: | ---: |
| 0% | 5.10 s | -2% | 0% |
| 10% | 4.60 s | 8% | 10% |
| 30% | 3.60 s | 28% | 30% |
| 50% | 2.60 s | 48% | 50% |
| 80% | 1.10 s | 78% | 80% |

An individual approved hit takes 0.10 s instead of 5 s: 50× faster, or a 98%
latency reduction. With 1,000 comparable requests and a 30% valid hit rate,
cache-first execution avoids 300 remote generations. Token charges may fall by
approximately 30% if cached and uncached requests have comparable billable token
counts; net savings must subtract decision, embedding and H100 hosting costs.

## Sensitivity to remote latency

Keep `D = 100 ms` and `h = 30%`:

| Assumed mean remote latency | Mean with JAC | Mean latency reduction | Break-even hit rate |
| --- | ---: | ---: | ---: |
| 2 s | 1.50 s | 25% | 5% |
| 5 s | 3.60 s | 28% | 2% |
| 10 s | 7.10 s | 29% | 1% |

Real JAC can bypass decision work on exact hits, memoized judgments and requests
that fail eligibility or expected-savings checks. The simplified calculation does
not include those optimizations or claim a measured production hit rate.

## Parallel mode and tail latency

In parallel mode, both paths start together. Approximately, a successful hit
returns after `min(D, G)` and a miss after `G`, plus scheduling overhead. With
the 5-second/100-ms assumptions and 30% valid hits, mean latency is about 3.53 s
(29.4% lower). All remote generations may start, so call avoidance and token-cost
savings cannot be inferred from the hit rate.

Mean savings do not establish p50/p95 savings. If misses remain among the slowest
requests, p95 can remain close to normal generation or increase in cache-first
mode. Measure the full distribution with rejected candidates, concurrency and
provider failures included. Correctness is a prerequisite: an incorrect reuse
does not count as a useful hit, regardless of how quickly it returns.
