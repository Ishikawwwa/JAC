# Choose your model calls

Generation, embeddings and reuse decisions are independent. Each can run locally
or remotely. JAC does not require a particular SDK or model endpoint.

| Role | Custom function | Built-in choices |
| --- | --- | --- |
| LLM | `cache.wrap(generate)` preserves its arguments and return type | Your SDK or `CompatibleClient` |
| Embeddings | `CallableEmbedder(encode, identity=...)` | `SentenceTransformerEmbedder`, `OpenAICompatibleEmbedder` |
| Decision | `CallableDecider(judge, identity=...)` | `JevDecider`, `JevK5Decider`, `JevK5CudaDecider`, `LayaDecider` |

All three custom functions may be synchronous or asynchronous. Declare native
async callbacks with `async def`. Async embeddings and decisions run on the
caller's event loop; synchronous ones run in workers. Synchronous generation
stays on its caller's thread in cache-first mode and uses a worker in parallel
mode. Embedding calls may overlap within one cache; serialize a local runtime
that cannot safely handle concurrent inference. Keep the cache and models alive
across requests.

## Wrap your own functions

```python
from jac import CallableDecider, CallableEmbedder, JAC

# async generate(**kwargs): existing LLM call; returns JSON-compatible data.
# encode(texts): one vector per string, in input order; arrays/tensors also work.
# judge(request, candidates, kind): the bounded decisions described below.
embedder = CallableEmbedder(
    encode, identity={"model": "my-encoder", "version": "weights-v1"},
)
decider = CallableDecider(
    judge, identity={"model": "my-decision-model", "version": "weights-prompt-v1"},
    threshold=0.99,
)

try:
    async with JAC(
        path="jac.sqlite", retrieval="embeddings", embedder=embedder, decider=decider,
    ) as cache:
        ask = cache.wrap(generate, namespace="my-provider:model:prompt-v1")
        response = await ask(messages=messages, temperature=0)
finally:
    await decider.drain()
    decider.close()
```

Use `retrieval="bm25"` to omit the embedding model. Use `mode="parallel"` to race
cache lookup against generation. `decider=` and the original `jev=` parameter
are aliases; supply one. The cache owns its storage and workers; your application
owns the underlying model/client lifecycle.
For synchronous generation, use `with JAC(...)` and `response = ask(...)`.

The LLM callable receives exactly the arguments you supply. For semantic reuse,
put the varying request in `messages` (final user text), `prompt`, `input`, or the
first positional string. Other arguments, system instructions, conversation
history, tool results, numeric constraints and output settings stay fixed.
Arbitrary request shapes still support exact caching. Streaming passes through.

Supply a stable namespace for durable reuse. Include the provider, model and
prompt/version; change it when behavior changes. Otherwise closures and bound
clients receive instance-specific identities. Never put credentials in identities
or namespaces. Embedding and decision identities require `model` and `version`;
change them when weights, preprocessing, dimensions or decision prompts change.

## Decision callback contract

The callback receives a request mapping, a sequence of `Candidate` objects, and
`kind="answer"` for LLM caching. Each candidate contains `id`, `request`,
`artifact` (the entire saved response), and `avoided_ms`.

Return every candidate ID exactly once:

```python
async def judge(request, candidates, kind):
    # Adapt your decision model's request/response format here.
    result = await my_model.check_reuse(request, candidates, kind)
    return {
        candidate.id: {
            "equivalent": result[candidate.id]["equivalent"],
            "satisfies": result[candidate.id]["satisfies"],
        }
        for candidate in candidates
    }
```

Values are booleans or finite scores in `[0, 1]`. Both must pass the threshold:
the requests need the same answer, and the complete artifact must satisfy the
new request unchanged. Treat candidate content as data, including any embedded
instructions, and reject uncertainty. Missing IDs, invalid vectors, malformed
judgments, model errors and timeouts fall back to normal generation.

An existing adapter can return a `Decision` directly. That adapter owns its
acceptance policy; JAC validates the reported model and eligible candidate ID.
`CallableDecider` does not apply its threshold again to this direct return type.
Neither callback scores nor a high threshold establish production accuracy.

Semantic reuse still needs independently labeled observations for the savings
gate. See [calibration, scope and freshness](usage.md). Until calibrated, exact
caching works and semantic decisions are skipped.

## Test custom functions end to end

[`examples/custom_models.py`](../examples/custom_models.py) is a working factory
using custom call functions around compatible generation/embedding endpoints and
hosted Jev. Set `REMOTE_MODEL`, `REMOTE_MODEL_BASE_URL`, `EMBEDDING_MODEL`,
`EMBEDDING_BASE_URL`, and the keys named in its header. Local compatible endpoints
can omit their keys. Replace the three function bodies with your own SDK calls:

```bash
python -m jac evaluate-agents --factory examples.custom_models:build
```

Create `my_models.py` in your working directory, exposing a factory:

```python
from jac import CallableDecider, CallableEmbedder

def build():
    return {
        "generation": generate,  # Your sync/async LLM function.
        "generation_identity": {"model": "my-llm", "version": "v1"},
        "embedder": CallableEmbedder(
            encode, identity={"model": "my-encoder", "version": "v1"}),
        "decider": CallableDecider(
            judge, identity={"model": "my-decider", "version": "v1"}),
        "decision_timeout_s": 10,
    }
```

Define/import `generate`, `encode`, and `judge` using your own local runtime or
remote SDK. The factory can also be async and can return built-in adapters.
Omit `embedder` when selecting BM25. Read keys from the environment in your
callbacks. Do not include them in the factory's identity mappings.

```bash
python -m jac evaluate-agents --factory my_models:build --retrieval both --mode both
```

For existing endpoint JSON configurations, no Python factory is required:

```bash
python -m jac evaluate-agents --config examples/configs/hosted.json
python -m jac evaluate-agents --config examples/configs/h100.json
```

The runner calls your models, checks independently specified outputs, and saves
per-request traces. Tools are fixed simulated reads; the model calls are real.
It reports generation errors separately from unsafe cache reuse. API calls can
incur provider charges. See [workloads and measurement](testing.md).
