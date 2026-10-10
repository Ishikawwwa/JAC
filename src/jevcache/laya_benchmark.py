"""Reproducible full JAC comparisons with real local Laya decision inference.

The generator is a delayed, cancellable fixture: end-to-end timings describe that
controlled scenario, not a measured live LLM service. Decision and embedding
inference, SQLite lookup, cache admission and cancellation are real.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import platform
import random
import sqlite3
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .call_cache import JAC, _semantic_request
from .decisions import Candidate, Decision, DecisionPolicy
from .laya_prompts import DEFAULT_PROMPT_VERSION, PROMPT_VERSIONS, laya_questions
from .laya_workload import workload
from .models import Context, fingerprint
from .providers import SentenceTransformerEmbedder, normalized
from .store import Entry, SQLiteStore

NAMESPACE = "laya-benchmark-v1"
FUNCTION_NAME = "controlled-generator-v1"
THRESHOLDS = (.5, .7, .8, .9, .95, .99, .995, .999, 1.0)
MODES = (
    "no_cache", "exact_only", "unsafe_bm25", "unsafe_embeddings",
    "laya_bm25_cache_first", "laya_embeddings_cache_first",
    "laya_bm25_parallel", "laya_embeddings_parallel",
)
JEVK5_MODES = tuple(name.replace("laya_", "jevk5_") for name in MODES)


def percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def timing(values):
    return {"count": len(values), "p50_ms": statistics.median(values) if values else None,
            "p95_ms": percentile(values, .95), "max_ms": max(values) if values else None}


def request_kwargs(text):
    return {"model": "synthetic-generator-v1", "temperature": 0,
            "messages": [
                {"role": "system", "content": "Follow the user's request precisely."},
                {"role": "user", "content": text},
            ]}


def call_request(text):
    return {"args": [], "kwargs": request_kwargs(text), "text": text}


def _identity():
    return fingerprint({"cache": NAMESPACE, "function": FUNCTION_NAME})


def _partition(text):
    call = {"args": [], "kwargs": request_kwargs(text)}
    return fingerprint({"function": _identity(),
                        "guard": fingerprint(_semantic_request(call)[1])})


def fixture_response(entry):
    # Keep the oracle outside model evidence. Human-readable challenge names
    # such as "wrong_negation" must not reveal the answer-validity label.
    return {"content": entry["answer"], "fixture_id": fingerprint(entry["id"])}


def make_entry(entry, avoided_ms):
    call = {"args": [], "kwargs": request_kwargs(entry["text"])}
    context = Context(NAMESPACE)
    key = fingerprint({"kind": "llm_answer", "function": _identity(), "call": call,
                       "context": context.to_dict(), "contract": "auto"})
    return Entry(key, "llm_answer", NAMESPACE, _partition(entry["text"]),
                 {"codec": "json-v1", "payload": fixture_response(entry)},
                 {**call, "text": entry["text"]}, {}, "1", "auto", time.time(),
                 None, avoided_ms, {"semantic_identity": None})


class DelayedGenerator:
    """The delay is synthetic; starts, completions and cancellation are observed."""

    def __init__(self, entries, queries, delay_ms):
        self.delay_ms = delay_ms
        self.entries = {entry["id"]: entry for entry in entries}
        self.oracle_ids = {fingerprint(entry["id"]): entry["id"] for entry in entries}
        self.responses = {entry["text"]: fixture_response(entry) for entry in entries}
        for query in queries:
            expected = query["expected_id"]
            self.responses[query["text"]] = (
                fixture_response(self.entries[expected]) if expected else
                {"content": "Synthetic generation result; answer correctness is not evaluated.",
                 "fixture_id": "generated:" + query["id"]})
        self.starts = self.completions = self.cancellations = 0

    async def __call__(self, **kwargs):
        self.starts += 1
        try:
            await asyncio.sleep(self.delay_ms / 1000)
        except asyncio.CancelledError:
            self.cancellations += 1
            raise
        self.completions += 1
        text = kwargs["messages"][-1]["content"]
        return dict(self.responses[text])


class UnsafeFirstCandidate:
    """Explicitly unsafe baseline: retrieval alone approves the first answer."""

    model = "unsafe-first-candidate-fixture"
    cache_identity = {"provider": "unsafe-retrieval-baseline", "version": "1"}

    async def decide(self, request, candidates, kind):
        return Decision(candidates[0].id if candidates else None, {}, self.model, 0)


def scored_acceptance(row, threshold):
    ranked = [(score, candidate) for candidate, score in row["scores"].items()
              if score >= threshold]
    return min(ranked, key=lambda item: (-item[0], item[1]))[1] if ranked else None


def select_threshold(rows):
    """Freeze a score cutoff using calibration labels, never held-out labels.

    Maximizes accepted valid answers among a fixed published cutoff grid, subject
    to no incorrect calibration reuse. This small-sample rule is experimental,
    not a production false-accept guarantee.
    """
    positives = sum(row["expected_id"] is not None for row in rows)
    comparisons = []
    for threshold in THRESHOLDS:
        accepted = [(row, scored_acceptance(row, threshold)) for row in rows]
        accepted = [(row, chosen) for row, chosen in accepted if chosen is not None]
        true_accepts = sum(chosen == row["expected_id"] for row, chosen in accepted)
        comparisons.append({
            "threshold": threshold, "true_accepts": true_accepts,
            "false_accepts": len(accepted) - true_accepts,
            "acceptance_precision": true_accepts / len(accepted) if accepted else None,
            "positive_recall": true_accepts / positives if positives else None,
        })
    qualified = [row for row in comparisons if row["false_accepts"] == 0
                 and row["true_accepts"] > 0]
    chosen = min(qualified, key=lambda row: (-row["true_accepts"], row["threshold"])) \
        if qualified else None
    return {"qualified": chosen is not None,
            "threshold": chosen["threshold"] if chosen else None,
            "rule": "max calibration true accepts, zero false accepts, fixed cutoff grid",
            "sensitivity": comparisons}


def summarize(rows):
    """Count quality on distinct cases; timing repeats are not independent labels."""
    per_query = {}
    for row in rows:
        per_query.setdefault(row["id"], []).append(row)
    positive_ids = {key for key, values in per_query.items()
                    if values[0]["label"] == "positive"}
    semantic_accepts = [row for row in rows if row["reuse_kind"] == "semantic"]
    true_accepts = [row for row in semantic_accepts if row["correct_reuse"]]
    false_accepts = [row for row in rows if row["reuse_kind"] != "none"
                     and not row["correct_reuse"]]
    accepted_ids = {row["id"] for row in semantic_accepts}
    true_ids = {row["id"] for row in true_accepts}
    false_ids = {row["id"] for row in false_accepts}
    true_ids -= false_ids
    # A query with inconsistent acceptance across repeats remains visible.
    inconsistent = [key for key, values in per_query.items() if len({
        (value["reuse_kind"], value["returned_fixture_id"]) for value in values}) > 1]
    return {
        "measurements": len(rows), "unique_queries": len(per_query),
        "end_to_end": timing([row["elapsed_ms"] for row in rows]),
        "decision": timing([value for row in rows for value in row["decision_ms"]]),
        "retrieval": timing([value for row in rows for value in row["retrieval_ms"]]),
        "by_label_latency": {label: timing([row["elapsed_ms"] for row in rows
                                           if row["label"] == label])
                             for label in sorted({row["label"] for row in rows})},
        "paths": dict(Counter(row["path"] for row in rows)),
        "decision_statuses": dict(Counter(row["decision_status"] for row in rows)),
        "generation_starts": sum(row["generation_starts"] for row in rows),
        "generation_completions": sum(row["generation_completions"] for row in rows),
        "generation_cancellations": sum(row["generation_cancellations"] for row in rows),
        "generation_starts_per_request": sum(row["generation_starts"] for row in rows)
        / len(rows) if rows else None,
        "runtime_decision_calls": sum(row["decision_calls"] for row in rows),
        "exact_reuse_measurements": sum(row["reuse_kind"] == "exact" for row in rows),
        "semantic_reuse_measurements": len(semantic_accepts),
        "true_semantic_reuse_measurements": len(true_accepts),
        "false_reuse_measurements": len(false_accepts),
        "semantic_acceptance_precision": len(true_accepts) / len(semantic_accepts)
        if semantic_accepts else None,
        "unique_semantic_accepts": len(accepted_ids), "unique_true_semantic_accepts": len(true_ids),
        "unique_false_reuses": len(false_ids),
        "unique_semantic_acceptance_precision": len(true_ids) / len(accepted_ids)
        if accepted_ids else None,
        "positive_semantic_recall": len(true_ids & positive_ids) / len(positive_ids)
        if positive_ids else None,
        "false_reuse_query_ids": sorted(false_ids), "inconsistent_query_ids": inconsistent,
        "quality_scope": "unchanged fixture answer reuse, not generated answer correctness",
    }


def _retrieve(store, text, strategy, embedder, candidate_limit):
    kwargs = {"kind": "llm_answer", "scope": NAMESPACE, "task": _partition(text),
              "text": text, "limit": candidate_limit}
    return store.embedding_candidates(**kwargs, embedder=embedder) \
        if strategy == "embeddings" else store.candidates(**kwargs)


def _seed(store, entries, avoided_ms, vectors, embedder):
    mapping = {}
    for source in entries:
        entry = make_entry(source, avoided_ms)
        store.put(entry)
        mapping[entry.key] = source["id"]
        if vectors is not None:
            with store.connection:
                store.connection.execute("INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?)", (
                    entry.key, fingerprint(embedder.cache_identity),
                    fingerprint(source["text"]), json.dumps(vectors[source["text"]])))
    return mapping


def _decider_stats(decider):
    return decider.stats() if callable(getattr(decider, "stats", None)) else {}


async def score_split(split, decider, embedder, candidate_limit, avoided_ms, vectors=None):
    """Score a corpus without choosing cutoffs or changing any model settings."""
    result = {}
    with SQLiteStore() as store:
        mapping = _seed(store, split["entries"], avoided_ms, vectors, embedder)
        # Indexing is excluded from measured lookup, and query inference is not cached.
        _retrieve(store, split["queries"][0]["text"], "embeddings", embedder, candidate_limit)
        for strategy in ("bm25", "embeddings"):
            rows = []
            for query in split["queries"]:
                started = time.perf_counter()
                entries = _retrieve(store, query["text"], strategy, embedder, candidate_limit)
                retrieval_ms = (time.perf_counter() - started) * 1000
                candidates = [Candidate(entry.key, entry.request, entry.value["payload"],
                                        entry.avoided_ms) for entry in entries]
                before = _decider_stats(decider)
                started = time.perf_counter()
                error = None
                try:
                    decision = await decider.decide(call_request(query["text"]), candidates,
                                                    "answer")
                    probabilities = dict(decision.probabilities)
                except Exception as exc:
                    error = type(exc).__name__
                    probabilities = {}
                elapsed_ms = (time.perf_counter() - started) * 1000
                after = _decider_stats(decider)
                scores = {}
                for candidate in candidates:
                    dimensions = laya_questions(
                        version=getattr(decider, "prompt_version", DEFAULT_PROMPT_VERSION)
                    )
                    values = [probabilities.get(candidate.id + ":" + dimension)
                              for dimension in dimensions]
                    if all(isinstance(value, (float, int)) and math.isfinite(value)
                           and 0 <= value <= 1 for value in values):
                        scores[mapping[candidate.id]] = min(values)
                rows.append({**query, "candidates": [mapping[entry.key] for entry in entries],
                             "scores": scores, "decision_ms": elapsed_ms if candidates else 0,
                             "retrieval_ms": retrieval_ms, "error": error,
                             "native_predict_calls": after.get("predict_calls", 0)
                             - before.get("predict_calls", 0),
                             "context_rejections": after.get("context_rejections", 0)
                             - before.get("context_rejections", 0),
                             "model_failures": after.get("failures", 0)
                             - before.get("failures", 0),
                             "worker_decision_ms": after.get("elapsed_ms", 0)
                             - before.get("elapsed_ms", 0)})
            result[strategy] = rows
    return result


def cutoff_metrics(rows, threshold):
    accepted = [(row, scored_acceptance(row, threshold)) for row in rows]
    accepted = [(row, chosen) for row, chosen in accepted if chosen is not None]
    true_ids = {row["id"] for row, chosen in accepted if chosen == row["expected_id"]}
    false_ids = {row["id"] for row, chosen in accepted if chosen != row["expected_id"]}
    positives = {row["id"] for row in rows if row["expected_id"] is not None}
    return {"threshold": threshold, "accepted": len(accepted),
            "true_accepts": len(true_ids), "false_accepts": len(false_ids),
            "acceptance_precision": len(true_ids) / len(accepted) if accepted else None,
            "recall": len(true_ids) / len(positives) if positives else None,
            "false_accept_query_ids": sorted(false_ids),
            "true_accept_query_ids": sorted(true_ids)}


def scoring_metrics(rows):
    inferred = [row for row in rows if row["native_predict_calls"] > 0]
    return {"queries": len(rows), "inferred_queries": len(inferred),
            "context_rejected_queries": sum(row["context_rejections"] > 0 for row in rows),
            "model_error_queries": sum(row["error"] is not None for row in rows),
            "no_candidate_queries": sum(not row["candidates"] for row in rows),
            "successful_inference_decision": timing([row["decision_ms"] for row in inferred
                                                      if row["error"] is None]),
            "successful_worker_decision": timing([row["worker_decision_ms"] for row in inferred
                                                    if row["error"] is None]),
            "all_lookup_decisions": timing([row["decision_ms"] for row in rows
                                             if row["candidates"]]),
            "context_rejection_latency": timing([row["decision_ms"] for row in rows
                                                  if row["context_rejections"]]),
            "retrieval": timing([row["retrieval_ms"] for row in rows])}


async def calibrate(split, decider, embedder, candidate_limit, avoided_ms, vectors=None):
    scored = await score_split(split, decider, embedder, candidate_limit, avoided_ms, vectors)
    result = {}
    for strategy, rows in scored.items():
        selected = select_threshold(rows)
        threshold = selected["threshold"]
        observations = [{"valid_hit": scored_acceptance(row, threshold) == row["expected_id"]
                         and row["expected_id"] is not None,
                         "decision_ms": row["decision_ms"]}
                        for row in rows] if threshold is not None else [
                            {"valid_hit": False, "decision_ms": row["decision_ms"]}
                            for row in rows]
        unsafe_observations = [{"valid_hit": bool(row["candidates"])
                                and row["candidates"][0] == row["expected_id"],
                                "decision_ms": 0.0} for row in rows]
        result[strategy] = {**selected, "queries": rows, "observations": observations,
                            "unsafe_observations": unsafe_observations,
                            "scoring_summary": scoring_metrics(rows),
                            "timings": {"decision": timing([row["decision_ms"] for row in rows
                                                             if row["candidates"]]),
                                        "retrieval": timing([row["retrieval_ms"]
                                                              for row in rows])}}
        print(f"Calibration {strategy}: threshold={threshold}, "
              f"valid hits={sum(row['valid_hit'] for row in observations)}/{len(rows)}",
              flush=True)
    return result


async def evaluate_held_out(split, calibration, decider, embedder, candidate_limit,
                            avoided_ms, vectors=None):
    """Shadow evaluation: held-out labels never select a deployment threshold."""
    scored = await score_split(split, decider, embedder, candidate_limit, avoided_ms, vectors)
    result = {}
    for strategy, rows in scored.items():
        frozen = calibration[strategy]["threshold"]
        result[strategy] = {
            "queries": rows, "scoring_summary": scoring_metrics(rows),
            "frozen_threshold": frozen,
            "frozen_configuration_quality": cutoff_metrics(rows, frozen)
            if frozen is not None else None,
            "semantic_configuration_qualified": frozen is not None,
            "exploratory_cutoff_diagnostics": [cutoff_metrics(rows, threshold)
                                                for threshold in THRESHOLDS],
            "notes": ["All held-out cases are scored, even when calibration disallows reuse.",
                      "Exploratory cutoffs do not tune the threshold or feed the savings gate.",
                      "Exact requests are included in this audit; normal exact hits skip Laya.",
                      "Worker timing includes preflight and parsing; it is not GPU kernel timing."],
        }
        print(f"Held-out shadow {strategy}: "
              f"inferred={result[strategy]['scoring_summary']['inferred_queries']}, "
              f"frozen threshold={frozen}", flush=True)
    return result


async def _wait_for_model(decider):
    drain = getattr(decider, "drain", None)
    if callable(drain):
        await drain()


async def _measure_query(query, cache, wrapped, generator, decider):
    started = time.perf_counter()
    response = await wrapped(**request_kwargs(query["text"]))
    elapsed_ms = (time.perf_counter() - started) * 1000
    trace = cache.last_trace
    started = time.perf_counter()
    await cache.drain()
    await _wait_for_model(decider)
    drain_ms = (time.perf_counter() - started) * 1000
    returned = generator.oracle_ids.get(response["fixture_id"], response["fixture_id"])
    reuse_kind = ("exact" if query["label"] == "exact" else "semantic") \
        if trace.path == "full_hit" else "none"
    return {**query, "elapsed_ms": elapsed_ms, "background_drain_ms": drain_ms,
            "path": trace.path, "reuse_kind": reuse_kind, "returned_fixture_id": returned,
            "correct_reuse": returned == query["expected_id"]
            if reuse_kind != "none" else None,
            "decision_status": trace.decision_status, "race_winner": trace.race_winner,
            "decision_calls": trace.jev_calls,
            "generation_starts": generator.starts,
            "generation_completions": generator.completions,
            "generation_cancellations": generator.cancellations,
            "decision_ms": [event.elapsed_ms for event in trace.events if event.kind == "jev"],
            "retrieval_ms": [event.elapsed_ms for event in trace.events
                             if event.name.startswith("retrieval_")],
            "trace": trace.to_dict()}


async def measure_mode(name, split, calibration, decider, embedder, vectors,
                       generation_ms, avoided_ms, repeats, candidate_limit, directory):
    strategy = "embeddings" if "embeddings" in name else "bm25"
    is_laya = name.startswith(("laya_", "jevk5_"))
    unsafe = name.startswith("unsafe_")
    selected = calibration[strategy]
    selected_decider = decider if is_laya and selected["qualified"] else (
        UnsafeFirstCandidate() if unsafe else None)
    if is_laya and selected["qualified"]:
        decider.threshold = selected["threshold"]
    observations = selected["unsafe_observations"] if unsafe else selected["observations"]
    base_path = directory / (name + "_base.sqlite")
    rows = []
    with SQLiteStore(base_path) as base:
        _seed(base, split["entries"], avoided_ms, vectors, embedder)
        for trial in range(repeats):
            queries = list(split["queries"])
            random.Random(42 + trial).shuffle(queries)
            for index, query in enumerate(queries):
                # Fresh identical stored corpus for every timed request: promotions,
                # miss admission and decision memoization cannot turn repeats exact.
                query_path = directory / f"{name}_{trial}_{index}.sqlite"
                with SQLiteStore(query_path) as store:
                    base.connection.backup(store.connection)
                    generator = DelayedGenerator(split["entries"], split["queries"],
                                                 generation_ms)
                    cache = JAC(store=store, namespace=NAMESPACE, ttl_s=None,
                                enabled=name != "no_cache", jev=selected_decider,
                                decision_policy=DecisionPolicy(min_samples=20, overhead_ms=0),
                                decision_timeout_s=30,
                                mode="parallel" if name.endswith("parallel") else "cache_first",
                                retrieval=strategy,
                                embedder=embedder if strategy == "embeddings" else None,
                                candidate_limit=candidate_limit)
                    wrapped = cache.wrap(generator, namespace=FUNCTION_NAME)
                    if selected_decider is not None:
                        # Transfer the *measured calibration* operation statistics to
                        # this request cohort. No held-out labels enter the gate.
                        for observation in observations:
                            wrapped.observe_reuse(request_kwargs(query["text"]), **observation)
                    try:
                        row = await _measure_query(query, cache, wrapped, generator, decider)
                        rows.append({**row, "repeat": trial})
                    finally:
                        await cache.drain()
                        await _wait_for_model(decider)
                        cache.close()
    result = {"summary": summarize(rows), "queries": rows,
              "semantic_enabled": selected_decider is not None,
              "threshold": selected["threshold"] if is_laya else None,
              "unsafe_baseline": unsafe}
    summary = result["summary"]
    print(f"{name}: p50={summary['end_to_end']['p50_ms']:.2f}ms "
          f"p95={summary['end_to_end']['p95_ms']:.2f}ms; "
          f"semantic={summary['unique_true_semantic_accepts']} true, "
          f"{summary['unique_false_reuses']} false; "
          f"generation starts={summary['generation_starts']}", flush=True)
    return result


def environment():
    packages = {}
    for package in ("jac-cache", "laya", "torch", "transformers", "sentence-transformers",
                    "tokenizers", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "cpu": platform.processor(), "sqlite": sqlite3.sqlite_version,
              "packages": packages}
    try:
        import torch

        result["torch_threads"] = torch.get_num_threads()
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["gpu"] = {"name": torch.cuda.get_device_name(),
                             "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                             "allocated_bytes": torch.cuda.memory_allocated(),
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    except ImportError:
        pass
    return result


def write_report(report, output):
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def run(*, model="convaiinnovations/laya", revision=None, device="cuda", dtype="auto",
              embedding_model="sentence-transformers/all-MiniLM-L6-v2",
              embedding_revision=None, repeats=3, generation_ms=500,
              candidate_limit=1, output=None, modes=MODES, decider=None, embedder=None,
              prompt_version=DEFAULT_PROMPT_VERSION):
    if repeats < 1 or not math.isfinite(generation_ms) or generation_ms <= 0:
        raise ValueError("repeats and generation_ms must be positive")
    if any(mode not in (*MODES, *JEVK5_MODES) for mode in modes) or not 1 <= candidate_limit <= 8:
        raise ValueError("invalid modes or candidate_limit")
    dataset = workload()
    started = time.perf_counter()
    if decider is None:
        from .laya import LayaDecider

        decider = LayaDecider(model=model, revision=revision, device=device, dtype=dtype,
                              threshold=1, max_candidates=candidate_limit,
                              prompt_version=prompt_version)
    preload = getattr(decider, "preload", None)
    if callable(preload):
        await preload()
    load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    if embedder is None:
        embedder = SentenceTransformerEmbedder(embedding_model, revision=embedding_revision,
                                               device="cpu")
    embedding_load_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    all_entries = dataset["calibration"]["entries"] + dataset["held_out"]["entries"]
    encoded = embedder.embed([entry["text"] for entry in all_entries])
    vectors = {entry["text"]: normalized(vector)
               for entry, vector in zip(all_entries, encoded, strict=True)}
    index_ms = (time.perf_counter() - started) * 1000
    # A real timed control run establishes the avoided fixture duration.
    generator = DelayedGenerator(dataset["held_out"]["entries"],
                                 dataset["held_out"]["queries"], generation_ms)
    control_ms = []
    for _ in range(3):
        started = time.perf_counter()
        await generator(**request_kwargs(dataset["held_out"]["queries"][0]["text"]))
        control_ms.append((time.perf_counter() - started) * 1000)
    avoided_ms = statistics.median(control_ms)
    # Exclude the first decision from calibration/timing summaries.
    first = dataset["calibration"]["entries"][0]
    entry = make_entry(first, avoided_ms)
    started = time.perf_counter()
    await decider.decide(call_request(first["text"]), [
        Candidate(entry.key, entry.request, entry.value["payload"], avoided_ms)], "answer")
    warmup_ms = (time.perf_counter() - started) * 1000
    warmup_model_stats = decider.stats(reset=True) \
        if callable(getattr(decider, "stats", None)) else None
    provider = decider.cache_identity.get("provider", "laya-local")
    provider_name = "JevK5" if provider == "jevk5-gguf" else "Laya"
    report: dict[str, Any] = {
        "schema_version": ("jac-jevk5-comparison-v1" if provider == "jevk5-gguf"
                           else "jac-laya-comparison-v1"), "dataset_version": dataset["version"],
        "dataset_sha256": fingerprint(dataset), "dataset": dataset,
        "model_identity": dict(decider.cache_identity),
        "embedding_identity": dict(embedder.cache_identity),
        "requested_device": device, "model_load_ms": load_ms, "model_warmup_ms": warmup_ms,
        "warmup_model_stats": warmup_model_stats,
        "embedding_load_ms": embedding_load_ms, "document_index_ms": index_ms,
        "generation": {"kind": "delayed-async-fixture", "configured_ms": generation_ms,
                       "measured_control": timing(control_ms), "avoided_ms": avoided_ms},
        "candidate_limit": candidate_limit, "repeats": repeats, "seed": 42,
        "environment": environment(), "runs": {},
        "notes": [
            f"Real {provider_name} and embedding inference; no generative LLM service was called.",
            "End-to-end speedups apply to the controlled delayed fixture only.",
            "Labels assess unchanged-answer reuse; generated answer correctness is not measured.",
            "Calibration and held-out task families are disjoint, including invalid-answer "
            "challenges.",
            "Thresholds use calibration only: zero false accepts and maximal recall.",
            "Measured calibration outcomes are transferred to held-out operation cohorts "
            "for the gate.",
            "Fresh identical corpus per timed request excludes promotion/memoization effects.",
            "Model loading, warmup, indexing and snapshot restoration are excluded from timing.",
            "Parallel execution counts generation starts even when the cache wins and cancels "
            "the fixture.",
            "Repeated timings do not multiply the number of independent correctness cases.",
            "Unsafe retrieval-only baselines are diagnostics, not recommended configurations.",
            "Variants run in fixed order; shuffled queries use seed 42 plus repeat index. "
            "Thermal or background-load differences can affect cross-variant timing.",
            "This small repository-authored synthetic suite has no independent label audit. "
            "Its descriptive p95 is not an estimate of production tail latency.",
        ],
    }
    report["calibration"] = await calibrate(dataset["calibration"], decider, embedder,
                                            candidate_limit, avoided_ms, vectors)
    report["calibration_model_stats"] = decider.stats(reset=True) \
        if callable(getattr(decider, "stats", None)) else None
    report["held_out_shadow"] = await evaluate_held_out(
        dataset["held_out"], report["calibration"], decider, embedder,
        candidate_limit, avoided_ms, vectors)
    report["held_out_shadow_model_stats"] = decider.stats(reset=True) \
        if callable(getattr(decider, "stats", None)) else None
    write_report(report, output)
    with tempfile.TemporaryDirectory(prefix="jac-laya-benchmark-") as tmp:
        for name in modes:
            result = await measure_mode(name, dataset["held_out"], report["calibration"],
                                        decider, embedder, vectors, generation_ms, avoided_ms,
                                        repeats, candidate_limit, Path(tmp))
            result["model_stats"] = decider.stats(reset=True) \
                if callable(getattr(decider, "stats", None)) else None
            report["runs"][name] = result
            write_report(report, output)
    report["final_environment"] = environment()
    report["completed"] = True
    write_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS,
                        default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--generation-ms", type=float, default=500)
    parser.add_argument("--candidate-limit", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--audit-only", action="store_true",
                        help="run calibration and held-out model audits without timed JAC variants")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark-results/laya-comparison.json"))
    args = parser.parse_args()
    if args.repeats < 1 or args.generation_ms <= 0:
        parser.error("repeats and generation-ms must be positive")
    options = vars(args)
    if options.pop("audit_only"):
        options["modes"] = ()
    report = asyncio.run(run(**options))
    print(f"Completed {len(report['runs'])} comparisons. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
