# Testing and measuring JAC

## Run the implementation tests

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -q
python -m ruff check src tests examples
```

Tests cover cache identity, scope/freshness, candidate rejection, persistence,
provider adapters, parallel races and cancellation. Optional decision runtimes
are injected, so the suite needs neither GPU weights nor API keys.

## Try the complete flow without tokens

```bash
python examples/cached_llm.py
python examples/parallel_llm.py
python -m jac evaluate --offline --repeats 3
python -m jac evaluate-agents --offline --repeats 3
```

Offline generation, decisions and embeddings are fixtures. Their measured delays
show execution and accounting, not live model latency or accuracy. Reports go to
ignored `benchmark-results/`; local metrics are not shipped with the repository.

The agent suite exercises support answers with tenant/knowledge-base revisions,
schema-bound invoice extraction, and portfolio summaries after tool reads. It
includes held-out paraphrases, changed numbers, output constraints, history,
tool results and simultaneous duplicate requests. Labels come from independent
expected outputs; they are not a production runtime validator. Loading, warmup,
seed requests and calibration are separate from measured requests.

Compare no cache, exact caching, BM25 and embedding retrieval in both execution
modes. Reports include response p50/p95, generation calls started, completions and
cancellations, decision/embedding/tool calls, response errors and unsafe reuse.
Calibration false positives (invalid reuse approved) and false negatives (valid
reuse rejected) are separate. Rejection reduces savings; it is not an incorrect
cached answer. Provider errors, incorrect responses, isolation failures or unsafe
calibration approvals give exit status 1. Conservative rejections alone do not.
Reports retain calibration scores and raw responses for diagnosing model failures.
Offline vectors and decisions are correctness fixtures with synthetic delays;
their timings do not estimate model performance. Fixed tools are simulated in
both offline and live runs. A small workload is a regression/evaluation aid,
not an accuracy certificate.

## Measure your remote provider

Use [the hosted configuration](../examples/configs/hosted.json), or follow
[JevK5 setup](jevk5.md) for the [H100 configuration](../examples/configs/h100.json).
Edit the model/endpoint and copy `examples/configs/.env.example` to
`examples/configs/.env`, then fill in the required keys:

```bash
python -m jac evaluate --config examples/configs/hosted.json --retrieval both --mode both --repeats 3
python -m jac evaluate --config examples/configs/h100.json --retrieval both --mode both --repeats 3
python -m jac evaluate-agents --config examples/configs/hosted.json --repeats 3
```

Run the command for your chosen setup. Live evaluations incur provider charges.
For arbitrary model SDKs/functions, use
`python -m jac evaluate-agents --factory my_models:build`; see the
[custom model guide](custom_models.md).
They compare no cache, exact caching and the two semantic retrieval modes with
separate calibration and the same seeded corpus. The small independent fact
oracle checks functionality; it does not certify production reuse accuracy.
Oracle or calibration failures give exit status 1 and a saved diagnostic report.

Evaluate your own labeled traffic before enabling semantic reuse. Include changed
constraints, stale/incorrect answers and rejected candidates. Report full response
p50/p95, calls started, tokens, valid hits and incorrect reuse. Keep loading,
warmup and calibration separate from warm latency; also measure concurrency.
Cancellation does not prove provider billing stopped. Timing repeats do not add
independent correctness cases, and zero errors on a small dataset do not establish
a production error rate. [Performance projections](performance.md) are assumptions
to test, not deployment measurements.
