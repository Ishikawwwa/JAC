# Using JAC

## Install and run without tokens

```bash
git clone https://github.com/Ishikawwwa/JAC.git
cd JAC
python -m pip install -e ".[dev]"
python examples/cached_llm.py
python examples/parallel_llm.py
python -m jac evaluate --offline
python -m unittest discover -s tests -q
```

The offline evaluation uses fixtures for generation, decisions and embeddings.
It verifies execution paths and prints latency, calls and correctness checks;
it is not a live model benchmark. Generated reports go to ignored
`benchmark-results/`.

## Wrap an existing remote call

JAC's base package has no third-party runtime dependencies. This example uses its
small OpenAI-compatible HTTP client; use your existing SDK if you prefer.

```python
import asyncio
import os
from jac import CompatibleClient, JAC

async def main():
    provider = CompatibleClient(
        model=os.environ["REMOTE_MODEL"],
        base_url=os.environ["REMOTE_MODEL_BASE_URL"],
        api_key=os.environ["REMOTE_MODEL_API_KEY"],
        timeout_s=120,
    )
    async with JAC(path="jac.sqlite", ttl_s=300) as cache:
        ask = cache.wrap(provider.complete, namespace="my-agent-v1")
        request = {"messages": [{"role": "user", "content": "Explain BM25."}]}
        first = await ask(**request)
        second = await ask(**request)  # Returns the stored response.
        print(second)
        print(cache.stats())

asyncio.run(main())
```

Set the three environment variables to your provider's model, base URL and key.
`examples/remote_llm.py` contains this runnable example. Sync functions use a normal
`with JAC(...)` block. Known Chat Completions, Responses and Anthropic Messages
SDK clients can use `cache.wrap_client(existing_client)`; install your own SDK.
Generic functions should return JSON-compatible values, or supply
`response_type=YourResponseModel` for durable typed reuse.

## Enable decision-validated reuse

```python
from jac import DecisionPolicy, JAC, JevDecider

cache = JAC(
    path="jac.sqlite",
    retrieval="bm25",
    candidate_limit=1,
    jev=JevDecider(threshold=0.99),  # Set TYPESAFE_API_KEY.
    decision_policy=DecisionPolicy(min_samples=20, overhead_ms=100),
)
ask = cache.wrap(existing_llm_call)
```

The decision model checks request equivalence and whether the complete saved
answer satisfies the new request unchanged. Similarity alone never approves a
semantic hit. Both checks must pass, along with deterministic validity checks.
The threshold requires workload-specific evaluation; it is not an error guarantee.

Exact matches bypass the decider. Semantic reuse additionally needs at least
20 comparable, independently verified observations for its expected-savings gate:

```python
# Repeat for labeled evaluation observations from this same configuration.
ask.observe_reuse(
    request_kwargs,
    valid_hit=independently_verified_hit,
    decision_ms=measured_total_decision_ms,
)
```

These names represent data from your evaluation, not model-generated labels.
With no observations, JAC keeps exact caching and normal generation. Collect
positive and negative cases, changed constraints and stale/incorrect artifacts;
only seed the deployment after its reuse error rate meets your application's
requirements. Model confidence is not a measured hit probability. The savings
gate compares observed valid-hit rate times avoided generation latency with
retrieval and decision overhead. Configuration changes require new observations.

For local decisions, see [JevK5 on H100 or GGUF](jevk5.md). Optional Laya uses
`python -m pip install -e ".[laya]"` and `LayaDecider(device="cuda")`.

## Choose BM25 or embeddings

BM25 requires no embedding model. For embeddings:

```bash
python -m pip install -e ".[embeddings]"
```

```python
from jac import JAC, JevDecider, SentenceTransformerEmbedder

cache = JAC(
    retrieval="embeddings",
    embedder=SentenceTransformerEmbedder(device="cpu"),
    jev=JevDecider(),
    candidate_limit=1,
)
```

MiniLM weights download on first use. Document vectors persist in SQLite; initial
indexing can cost more than warm lookups. Vector search scans the eligible cache
partition and targets small caches rather than millions of records. Both retrieval
strategies use the same decision and validity checks. Only final user text may vary
for semantic candidates; history, settings and numeric constraints are guarded.

## Parallel execution

```python
cache = JAC(path="jac.sqlite", mode="parallel", jev=decider)
ask = cache.wrap(existing_llm_call)
response = await ask(**request)
# At application shutdown, after all requests finish:
await cache.drain()
await decider.drain()  # For a local decider with background inference.
cache.close()
decider.close()
```

Use `drain_sync()` for a sync application. Context-manager exit drains the cache
automatically. Keep it alive across requests; draining after every request can
wait for losing work. Generation starts speculatively and can remain billable
after cancellation. Running sync functions and CUDA kernels cannot be forcibly
stopped by canceling their awaiter. Streaming requests pass through.

## Scope, freshness and validation

```python
from jac import Context, JAC

cache = JAC(
    path="jac.sqlite", namespace="support-v1", ttl_s=300,
    context=Context(scope="tenant:42", dependencies={"kb": "revision-7"}),
    validator=my_answer_validator,
)
```

`my_answer_validator(request_mapping, response)` returns a boolean and may be
async. Request mappings contain `args` and `kwargs`. Supply scope per tenant and
dependency revisions for changing data. JAC does not discover external changes or
verify that the original generated answer is true. SQLite stores request/response
content; manage its access and retention. Hosted deciders receive candidate
requests and responses. Cache I/O failures fall back to normal generation.

Inspect `cache.last_trace` and `cache.stats()` for actual calls and hits. Cached
SDK responses retain original IDs and usage fields; those are not new billing
records. Concurrent identical requests coalesce within one JAC instance.

## Run live measurements

Copy `examples/configs/.env.example` to `examples/configs/.env`. For hosted Jev, fill `OPENAI_API_KEY` and
`TYPESAFE_API_KEY`, then edit `examples/configs/hosted.json` for your endpoints and models:

```bash
python -m jac evaluate --config examples/configs/hosted.json --retrieval bm25 --mode cache_first
python -m jac evaluate --config examples/configs/hosted.json --retrieval both --mode both
```

For an H100 decider plus any remote compatible generator, edit the generation
model/base URL in `examples/configs/h100.json`, set `REMOTE_MODEL_API_KEY`, and follow
[JevK5 setup](jevk5.md). The runner loads an adjacent `.env` without overriding
existing environment variables. Live commands incur provider charges. The small
fact workload has independent labels and separate calibration; it is a functional
evaluation, not a production accuracy certificate. See [evaluation](testing.md).
