"""Measure a real local encoder and persistent vector retrieval; no LLM/API keys.

Install: python -m pip install -e ".[embeddings]"
Run:     python examples/local_embeddings_smoke.py
"""

import json
import tempfile
import time
from pathlib import Path

from jac import SentenceTransformerEmbedder
from jevcache.store import Entry, SQLiteStore


def main():
    started = time.perf_counter()
    encoder = SentenceTransformerEmbedder()
    load_ms = (time.perf_counter() - started) * 1000
    # Warm inference is separate from retrieval timings.
    started = time.perf_counter()
    encoder.embed(["warmup"])
    warmup_ms = (time.perf_counter() - started) * 1000
    query = "automobile"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vectors.sqlite"
        with SQLiteStore(path) as store:
            for key, text in (("vehicle", "car"), ("fruit", "banana")):
                store.put(Entry(key, "llm_answer", "demo", "topic", text, {"text": text},
                                {}, "v1", "text", time.time(), None, 1000))
            bm25 = store.candidates(kind="llm_answer", scope="demo", task="topic", text=query)
            rows.append({"strategy": "bm25", "candidate_ids": [entry.key for entry in bm25]})
        # Reopen for each pass to verify document vectors survive a store restart.
        for phase in ("first_lookup_with_document_encoding", "persisted_vectors_lookup"):
            with SQLiteStore(path) as store:
                calls = encoder.calls
                started = time.perf_counter()
                results = store.embedding_candidates(kind="llm_answer", scope="demo", task="topic",
                    text=query, embedder=encoder, limit=1)
                elapsed = (time.perf_counter() - started) * 1000
                if not results or results[0].key != "vehicle":
                    raise AssertionError("Local encoder did not rank car first for automobile")
                rows.append({"strategy": "embeddings", "phase": phase,
                             "candidate_ids": [entry.key for entry in results],
                             "retrieval_ms": elapsed, "encoding_batches": encoder.calls - calls})
    report = {"real_local_encoder": encoder.cache_identity, "model_load_ms": load_ms,
              "warmup_ms": warmup_ms, "results": rows,
              "note": "Retrieval smoke test only; no Jev/generation or production accuracy claim."}
    output = Path("benchmark-results/local-embeddings-smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
