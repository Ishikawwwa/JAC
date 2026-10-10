"""Measure real local retrieval on realistic, hand-labeled LLM requests."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from .call_cache import _semantic_request
from .models import fingerprint
from .providers import SentenceTransformerEmbedder, normalized
from .realistic_workload import FAMILIES, requests
from .store import Entry, SQLiteStore


def guard(text):
    call = {"args": [], "kwargs": {"model": "benchmark-model", "temperature": 0,
            "messages": [{"role": "system", "content": "Follow the user's request precisely."},
                         {"role": "user", "content": text}]}}
    return fingerprint(_semantic_request(call)[1])


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


class RecordingEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.cache_identity = encoder.cache_identity
        self.vectors = {}
        self.batches = 0
        self.texts_encoded = 0

    def embed(self, texts):
        # Record vectors for analysis without caching query inference. Every
        # measured embedding retrieval still performs a real encoder call.
        vectors = self.encoder.embed(texts)
        self.batches += 1
        self.texts_encoded += len(texts)
        self.vectors.update(zip(texts, vectors, strict=True))
        return vectors


def summarize(rows):
    positives = [row for row in rows if row["label"] == "positive"]
    negatives = [row for row in rows if row["label"] != "positive"]
    latencies = [ms for row in rows for ms in row["retrieval_ms"]]
    return {
        "positive_queries": len(positives), "negative_queries": len(negatives),
        "positive_recall_at_1": sum(row["expected_id"] in row["candidates"][:1]
                                    for row in positives) / len(positives),
        "positive_recall_at_5": sum(row["expected_id"] in row["candidates"]
                                    for row in positives) / len(positives),
        "negatives_with_candidates": sum(bool(row["candidates"]) for row in negatives),
        "measurements": len(latencies), "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, .95), "max_ms": max(latencies),
        "positive_failures_at_1": [row["id"] for row in positives
                                   if row["expected_id"] not in row["candidates"][:1]],
    }


def thresholds(rows):
    # Diagnostic only: never used to authorize reuse in JAC. Labels are hand-written,
    # and this suite is too small to choose a production threshold.
    result = []
    for threshold in (.8, .85, .9, .95):
        accepted = [row for row in rows if row.get("top_cosine", -1) >= threshold]
        correct = [row for row in accepted
                   if row["expected_id"] is not None
                   and row["candidates"][0] == row["expected_id"]]
        result.append({"threshold": threshold, "accepted": len(accepted),
                       "correct_accepts": len(correct),
                       "incorrect_accepts": len(accepted) - len(correct),
                       "precision": len(correct) / len(accepted) if accepted else None,
                       "positive_recall": len(correct) / sum(
                           row["label"] == "positive" for row in rows)})
    return result


def run(*, model="sentence-transformers/all-MiniLM-L6-v2", repeats=3, revision=None):
    started = time.perf_counter()
    encoder = RecordingEncoder(SentenceTransformerEmbedder(model, revision=revision, device="cpu"))
    load_ms = (time.perf_counter() - started) * 1000
    queries = requests()
    report = {
        "model": encoder.cache_identity, "device": "cpu", "model_load_ms": load_ms,
        "cache_entries": len(FAMILIES), "unique_queries": len(queries), "repeats": repeats,
        "labels": dict(Counter(row["label"] for row in queries)),
        "categories": dict(Counter(family[1] for family in FAMILIES)),
        "notes": ["Real embedding inference and actual SQLite retrieval; no simulated delays.",
                  "No Jev, decision LLM or generative LLM calls were made.",
                  "Hand-written labels measure request retrieval, not answer correctness.",
                  "Latency includes query encoding, SQLite reads and cosine ranking.",
                  "Model loading, warmup and initial document indexing are separate.",
                  "Raw search measures recall; guarded search also applies JAC request guards.",
                  "A retrieved candidate is not permission to reuse its answer."],
        "runs": {},
    }
    started = time.perf_counter()
    encoder.embed(["Warm the local request encoder."])
    report["warmup_ms"] = (time.perf_counter() - started) * 1000
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "realistic.sqlite"
        canonical_text = {family[0]: family[2] for family in FAMILIES}
        guards = {key: guard(text) for key, text in canonical_text.items()}
        with SQLiteStore(path) as store:
            for key, text in canonical_text.items():
                store.put(Entry(key, "llm_answer", "agent", "requests", None, {"text": text},
                                {}, "v1", "auto", time.time(), None, 1000))
            started = time.perf_counter()
            store.embedding_candidates(kind="llm_answer", scope="agent", task="requests",
                                       text=queries[0]["text"], embedder=encoder, limit=5)
            report["initial_indexing_and_lookup_ms"] = (time.perf_counter() - started) * 1000
        # Reopen after indexing so the benchmark exercises persistent vectors.
        with SQLiteStore(path) as store:
            if not store.has_fts:
                raise RuntimeError("This BM25 comparison requires SQLite FTS5")
            for guarded in (False, True):
                measured = {strategy: {row["id"]: {**row, "retrieval_ms": []} for row in queries}
                            for strategy in ("bm25", "embeddings")}
                for trial in range(repeats):
                    shuffled = list(queries)
                    random.Random(42 + trial).shuffle(shuffled)
                    for query in shuffled:
                        query_guard = guard(query["text"])
                        eligible = ((lambda entry, g=query_guard: guards[entry.key] == g)
                                    if guarded else None)
                        strategies = ("bm25", "embeddings") if trial % 2 == 0 else (
                            "embeddings", "bm25")
                        for strategy in strategies:
                            started = time.perf_counter()
                            kwargs = dict(kind="llm_answer", scope="agent", task="requests",
                                          text=query["text"], limit=5, eligible=eligible)
                            candidates = (store.candidates(**kwargs) if strategy == "bm25" else
                                          store.embedding_candidates(**kwargs, embedder=encoder))
                            elapsed = (time.perf_counter() - started) * 1000
                            row = measured[strategy][query["id"]]
                            row["retrieval_ms"].append(elapsed)
                            row["candidates"] = [entry.key for entry in candidates]
                            if strategy == "embeddings" and candidates:
                                qvec = normalized(encoder.vectors[query["text"]])
                                dvec = normalized(
                                    encoder.vectors[canonical_text[candidates[0].key]])
                                row["top_cosine"] = sum(
                                    a * b for a, b in zip(qvec, dvec, strict=True))
                for strategy in ("bm25", "embeddings"):
                    rows = list(measured[strategy].values())
                    report["runs"][strategy + ("_guarded" if guarded else "_raw")] = {
                        "summary": summarize(rows), "queries": rows,
                        "similarity_only_diagnostic": thresholds(rows)
                        if strategy == "embeddings" else None,
                    }
    report["encoding_batches"] = encoder.batches
    report["texts_encoded"] = encoder.texts_encoded
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--revision")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark-results/realistic-retrieval.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = run(model=args.model, repeats=args.repeats, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REAL encoder on CPU: {report['cache_entries']} saved requests, "
          f"{report['unique_queries']} new requests, {args.repeats} timing repeats.")
    print(f"{'Mode':<22} {'Recall@1':>10} {'Recall@5':>10} {'p50 ms':>10} {'p95 ms':>10}")
    for name, run_result in report["runs"].items():
        summary = run_result["summary"]
        print(f"{name:<22} {summary['positive_recall_at_1']:10.1%} "
              f"{summary['positive_recall_at_5']:10.1%} {summary['p50_ms']:10.2f} "
              f"{summary['p95_ms']:10.2f}")
    print("Similarity-only diagnostics (guarded; does NOT authorize reuse):")
    for row in report["runs"]["embeddings_guarded"]["similarity_only_diagnostic"]:
        print(f"  cosine >= {row['threshold']}: correct accepts={row['correct_accepts']}, "
              f"incorrect accepts={row['incorrect_accepts']}")
    print(f"Full results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
