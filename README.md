# JAC — Jev As Cache

**Return a saved answer instead of generating it again.** JAC wraps your existing
LLM call: a valid cache hit returns the stored response; a miss runs the original
call and saves its result. Keep your agent and provider interface.

- **Skip generation:** exact hits need no decision model or API call.
- **Reuse equivalent requests:** retrieve candidates with BM25 or embeddings,
  then ask Jev or a local decision model whether an answer can be returned unchanged.
- **Protect miss latency:** optional parallel mode races lookup against generation.

## Quick start

```bash
git clone https://github.com/Ishikawwwa/JAC.git
cd JAC
python -m pip install -e .
```

Wrap an existing async function (sync functions work too):

```python
from jac import JAC

async with JAC(path="jac.sqlite", ttl_s=300) as cache:
    ask = cache.wrap(existing_llm_call)
    response = await ask(model="your-model", messages=messages)
    repeated = await ask(model="your-model", messages=messages)  # No generation.
```

Or keep your supported provider client's API with `cache.wrap_client(existing_client)`.

## Add semantic reuse

```python
from jac import JAC, JevDecider

cache = JAC(retrieval="bm25", jev=JevDecider())  # TYPESAFE_API_KEY
ask = cache.wrap(existing_llm_call)
```

Choose `retrieval="embeddings"` and supply an `embedder` for vector retrieval.
Use `CallableEmbedder` and `CallableDecider` for your own sync/async local or remote
functions. [Model selection and callback contracts](docs/custom_models.md).
Local **JevK5** and **Laya** adapters are also available. Semantic decisions require
labeled calibration observations; until then JAC uses exact caching.
See the [quickstart](docs/usage.md) and [H100 / JevK5 setup](docs/jevk5.md).

For a race, use `JAC(mode="parallel", ...)`. A miss continues generating without
waiting for lookup. Generation may already be billable on a hit; cancellation
does not guarantee a refund. Drain background work at shutdown.

## What could you save?

**Illustrative H100 deployment:** assume a remote LLM call averages **5 seconds**
and retrieval plus decision checks take **100 ms**. An approved semantic hit then
returns **50× faster**. At a **30% valid hit rate**, cache-first execution reduces
average latency **28%** and generation calls **30%**.

This is a calculation, not a measured JAC benchmark or an industry-wide latency
average. JevK5's author reports approximately 13 ms per short H100 decision using
its optimized CUDA runtime; total cache latency includes additional work.
[Assumptions and calculations](docs/performance.md).

## Try it

Runnable examples live in `examples/`; hosted and H100 configs are in
`examples/configs/`. Setup and measurement guides are in `docs/`.

```bash
python -m jac evaluate --offline     # Token-free functionality fixtures.
python -m jac evaluate-agents --offline  # Support, extraction and tool-informed workflows.
python -m unittest discover -s tests -q
```

JAC v0.1 is experimental. Use scope, dependency revisions, TTLs and application
validation for your data. Decision confidence alone does not establish correct
reuse. Streaming and unsupported response shapes pass through to the provider.

[Usage](docs/usage.md) · [Evaluation](docs/testing.md) · [MIT license](LICENSE)
