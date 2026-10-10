# JevK5 decisions: H100 or GGUF

JAC can use [JevK5](https://github.com/allebee/jevk5), an open-weight decision model,
to validate reusable responses. It does not generate the answer returned to the
user. Exact matches bypass the model. Semantic reuse needs calibration observations
and the same scope/freshness policies as other JAC deciders.

## H100 with the optimized CUDA runtime

Use Linux, Python 3.12 and a CUDA-compatible PyTorch installation in a separate
environment. The upstream runtime needs recent Torch/Transformers versions and
optionally flash-linear-attention. Install JAC and the pinned runtime:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,embeddings]"
python -m pip install "jevk5[fast] @ git+https://github.com/allebee/jevk5@f26426d16f59e8bbe1470e5b162cc89329e29b29"
```

Then wrap your existing remote generation function:

```python
from jac import DecisionPolicy, JAC, JevK5CudaDecider

decider = JevK5CudaDecider(
    model="alibiserikbay/JevK5",
    revision="c4f7fdb3aeab5582336406e78d3bef11bf98833d",
    device="cuda", graphs=True, temperature=1.22, threshold=0.99,
)
await decider.preload()  # Download, load bf16 weights and capture CUDA graphs.
cache = JAC(
    path="jac.sqlite", jev=decider, retrieval="bm25", candidate_limit=1,
    decision_policy=DecisionPolicy(min_samples=20, overhead_ms=100),
)
ask = cache.wrap(existing_remote_llm_call, namespace="my-agent-v1")
response = await ask(**request)
# At application shutdown:
await cache.drain()
await decider.drain()
cache.close()
decider.close()
```

`preload()` is outside request timing. Upstream graph capture has a substantial
startup cost; reuse one long-lived decider. The 4B bf16 model requires roughly
9 GB VRAM plus runtime/graph buffers according to upstream. Two reuse questions
are evaluated separately for each candidate, with zero generated output tokens.
Oversized evidence, invalid option probabilities or inference failures cannot
approve reuse. Async cancellation stops remaining checks, but a running GPU
operation may finish before `drain()` returns.

This adapter has protocol and lifecycle tests using injected runtimes. JAC has
not measured an H100 deployment. [Upstream H100 inference measurements](https://github.com/allebee/jevk5#results-on-jevbenchs-public-items)
use its optimized runtime; they are not end-to-end JAC latency guarantees.
[Performance projections](performance.md) make the deployment assumptions explicit.

## Run with a remote generator

Edit `examples/configs/h100.json` to supply your remote model ID and API base URL. Copy
`examples/configs/.env.example` to `examples/configs/.env` and fill `REMOTE_MODEL_API_KEY`. No hosted decision token
is required when using the local CUDA adapter.

```bash
python -m jac evaluate --config examples/configs/h100.json --retrieval bm25 --mode cache_first
python -m jac evaluate --config examples/configs/h100.json --retrieval both --mode both
```

The generator is real; calls consume provider credits. The evaluation uses a small
labeled fact workload to seed its own profile. Production semantic reuse still
requires your independently verified observations via `ask.observe_reuse(...)`.
Until calibrated, JAC uses exact caching. A threshold of 0.99 is a conservative
starting setting, not a measured guarantee that reuse is correct.

## GGUF through llama.cpp

For a smaller GPU or a separate server, install only the upstream prompt package:

```bash
python -m pip install -e ".[embeddings]"
python -m pip install --no-deps "jevk5 @ git+https://github.com/allebee/jevk5@f26426d16f59e8bbe1470e5b162cc89329e29b29"
```

Download `jevk5-4b-v0.3-Q8_0.gguf` from the
[pinned model revision](https://huggingface.co/alibiserikbay/JevK5-GGUF/tree/ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a)
and install a compatible [llama.cpp](https://github.com/ggml-org/llama.cpp/releases)
binary (the adapter's wire contract targets b11429). Start a private server:

```bash
llama-server -m /path/to/jevk5-4b-v0.3-Q8_0.gguf -c 8192 -ngl 99 -np 1 --host 127.0.0.1 --port 8095 --no-context-shift --cache-ram 0 --ctx-checkpoints 0
```

```python
from jac import JevK5Decider

decider = JevK5Decider(
    base_url="http://127.0.0.1:8095",
    revision="ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a",
    temperature=1.22, threshold=0.99,
)
```

Use this decider in the same `JAC(...)` constructor and calibrate it independently.
The GGUF adapter reads one bounded answer token per question and checks complete
token accounting. Its precision, runtime and performance differ from the H100
bf16 path. The H100 upstream figures should not be attributed to llama.cpp.

Both backends support BM25 or embeddings, cache-first or parallel execution, and
the optional versioned reuse prompts. Prompt names beginning `laya-reuse-*` select
shared wording, not a particular model. See [testing and evaluation](testing.md).
