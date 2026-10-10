# Evaluate your deployment

JAC ships workload definitions and runners, not a universal production latency
or accuracy result. [PERFORMANCE.md](PERFORMANCE.md) contains an explicitly
modeled H100/remote-LLM scenario. Measure your actual provider and traffic before
using those projections for capacity planning.

## Token-free functionality

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -q
python -m ruff check src tests examples
python -m jac evaluate --offline
python -m jac benchmark --requests 64 --output benchmark-results/report.json
```

Offline evaluations use generation/decision fixtures; reported delays are not
live GPU or API timings. Unit tests cover cache identity, scope/freshness,
candidate rejection, parallel races, cancellation, persistence and provider
contracts. Optional heavy dependencies are replaced with injected runtimes.

## Real decisions and remote generation

Follow [JEVK5.md](JEVK5.md) for an H100 CUDA decider. Edit the placeholder remote
model and endpoint in `jac.remote.json`, then set `REMOTE_MODEL_API_KEY` in an
adjacent `.env`:

```bash
python -m jac evaluate --config jac.remote.json --retrieval both --mode both --repeats 3 --output benchmark-results/remote.json
```

Alternatively use hosted Jev with `jac.cloud.json`. These commands consume provider
credits. Every variant starts with the same seeded corpus; setup and independent
calibration are reported separately from timed requests. The small fact workload
checks real wiring and answer correctness; a failed oracle or calibration gives
exit status 1 and a saved diagnostic report. A strict oracle can reject alternate
correct wording. Inspect traces rather than interpreting every quality failure
as a broken installation. Do not export this workload's labels as production
calibration.

## Broader request and retrieval experiments

```bash
python -m pip install -e ".[embeddings]"
python examples/local_embeddings_smoke.py
python -m jevcache.realistic_benchmark --repeats 3
```

The realistic workload includes coding, support, writing and operations requests,
paraphrases, changed requirements and unrelated requests. It compares real MiniLM
retrieval with BM25. Retrieval accuracy alone does not authorize answer reuse.

For GGUF JevK5 decision audits and the full JAC runtime with a controlled
generation fixture:

```bash
python -m jevcache.jevk5_benchmark --base-url http://127.0.0.1:8095 --prompt-version laya-reuse-v3-bindings --generation-ms 5000 --repeats 3 --output benchmark-results/jevk5-comparison.json
python examples/jevk5_reuse_demo.py --calibration benchmark-results/jevk5-comparison.json
```

The full comparison separates calibration from evaluation, restores the same
corpus per request and includes exact-only, BM25/embedding and parallel/cache-first
variants. Its generator is a timed fixture; only decisions and retrieval are live.
The short reuse demo selects two positive cases and a negative control, so it is
a functionality demonstration rather than an independent accuracy benchmark.

Laya's analogous runner is `python -m jevcache.laya_benchmark`. Optional prompt
replay commands require a prior audit produced by that runner; supply its path
with `--source` rather than expecting a bundled report. All generated reports
belong in ignored `benchmark-results/`.

## Report the useful metrics

- Full response p50/p95, including rejected decisions and misses; separately
  report time to first token if the application streams.
- Generation starts, completions, tokens, decision passes and embedding calls.
  A cancellation request does not establish avoided provider billing.
- Independently labeled valid hits and incorrect reuse, using distinct requests.
  Timing repeats do not multiply the number of correctness cases.
- Loading, graph capture, warmup, indexing and calibration outside warm latency,
  with cold-start costs reported separately.
- Hardware, model/runtime revisions, precision, dataset identity, candidate count
  and load/concurrency. Single-request timings do not establish throughput.

Compare no cache, exact-only cache and both semantic retrieval strategies on the
same requests. Include candidate rejections and verify source freshness. If the
savings gate skips decisions, report that behavior rather than claiming it proves
decision-model accuracy. Zero observed errors on a small dataset do not establish
a production error rate.
