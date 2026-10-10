import tempfile
import unittest
from pathlib import Path

from jac import JAC, DecisionPolicy
from jevcache.evaluate import FixtureClient, FixtureDecider, FixtureEmbedder, _request
from jevcache.models import fingerprint
from jevcache.store import Entry, SQLiteStore


def entry(key, text, *, dependencies=None, scope="test", expires=1000):
    return Entry(key, "llm_answer", scope, "partition", {"payload": text},
                 {"text": text}, dependencies or {}, "v1", "auto", 0, expires, 200)


class RecordingEmbedder:
    cache_identity = {"model": "test-v1"}

    def __init__(self):
        self.batches = []

    def embed(self, texts):
        self.batches.append(texts)
        vectors = {"car": [1, 0], "automobile": [1, 0], "banana": [0, 1]}
        return [vectors[text] for text in texts]


class RetrievalTests(unittest.TestCase):
    def test_configuration(self):
        for kwargs in ({"retrieval": "hybrid"}, {"retrieval": "embeddings"},
                       {"candidate_limit": 0}, {"candidate_limit": True}):
            with self.assertRaises(ValueError):
                JAC(path=":memory:", **kwargs)

    def test_embeddings_persist_and_filter_before_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.sqlite"
            encoder = RecordingEmbedder()
            with SQLiteStore(path, clock=lambda: 10) as store:
                store.put(entry("wrong-revision", "car", dependencies={"source": "old"}))
                store.put(entry("good", "car", dependencies={"source": "new"}))
                store.put(entry("expired", "car", expires=5))
                store.put(entry("wrong-tenant", "banana", scope="other"))
                kwargs = dict(kind="llm_answer", scope="test", task="partition",
                              text="automobile", embedder=encoder, limit=1,
                              eligible=lambda e: e.dependencies == {"source": "new"})
                self.assertEqual([e.key for e in store.embedding_candidates(**kwargs)], ["good"])
                self.assertEqual(encoder.batches, [["automobile", "car"]])
            with SQLiteStore(path, clock=lambda: 10) as store:
                self.assertEqual(store.embedding_candidates(**kwargs)[0].key, "good")
                self.assertEqual(encoder.batches[-1], ["automobile"])
                encoder.cache_identity = {"model": "test-v2"}
                store.embedding_candidates(**kwargs)
                self.assertEqual(encoder.batches[-1], ["automobile", "car"])
                store.invalidate(scope="test")
                self.assertEqual(store.connection.execute(
                    "SELECT COUNT(*) FROM embeddings").fetchone()[0], 0)

    def test_bm25_filters_before_top_k(self):
        with SQLiteStore(clock=lambda: 10) as store:
            store.put(entry("bad", "car car car", dependencies={"v": "old"}))
            store.put(entry("good", "car", dependencies={"v": "new"}))
            rows = store.candidates(kind="llm_answer", scope="test", task="partition",
                text="car", limit=1, eligible=lambda e: e.dependencies == {"v": "new"})
            self.assertEqual([e.key for e in rows], ["good"])

    def test_invalid_vectors_fail_closed(self):
        encoder = RecordingEmbedder()
        with SQLiteStore(clock=lambda: 10) as store:
            store.put(entry("key", "car"))
            for vector in ([0, 0], [float("nan"), 1], [True, 1], [], ["1"]):
                encoder.embed = lambda texts, v=vector: [v for _ in texts]
                with self.assertRaises(ValueError):
                    store.embedding_candidates(kind="llm_answer", scope="test", task="partition",
                                               text="car", embedder=encoder)


class WrapperRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_strategies_and_modes(self):
        for strategy in ("bm25", "embeddings"):
            for mode in ("cache_first", "parallel"):
                with self.subTest(strategy=strategy, mode=mode):
                    encoder = FixtureEmbedder()
                    client = FixtureClient()
                    async with JAC(path=":memory:", retrieval=strategy, embedder=encoder,
                        mode=mode, jev=FixtureDecider(),
                        decision_policy=DecisionPolicy(1, 0)) as cache:
                        wrapped = cache.wrap(client.complete)
                        first = _request("What is the capital of France?")
                        await wrapped(**first)
                        await cache.drain()
                        before = encoder.calls
                        await wrapped(**first)
                        await cache.drain()
                        self.assertEqual(encoder.calls, before)  # Exact never embeds.
                        different = _request("Name the French capital city.")
                        wrapped.observe_reuse(different, valid_hit=True, decision_ms=1)
                        response = await wrapped(**different)
                        self.assertIn("Paris", response["choices"][0]["message"]["content"])
                        self.assertEqual(cache.last_trace.path, "full_hit")
                        self.assertEqual(cache.last_trace.llm_calls, mode == "parallel")
                        await cache.drain()

    async def test_embedding_failure_generates_normally(self):
        encoder = FixtureEmbedder()
        async with JAC(path=":memory:", retrieval="embeddings", embedder=encoder,
                       jev=FixtureDecider(), decision_policy=DecisionPolicy(1, 0)) as cache:
            wrapped = cache.wrap(FixtureClient().complete)
            await wrapped(**_request("What is the capital of France?"))
            request = _request("Name the French capital city.")
            wrapped.observe_reuse(request, valid_hit=True, decision_ms=0)
            encoder.embed = lambda _: (_ for _ in ()).throw(RuntimeError("Unavailable"))
            await wrapped(**request)
            self.assertEqual(cache.last_trace.path, "miss")
            self.assertEqual(cache.last_trace.llm_calls, 1)

    async def test_profiles_are_specific_to_retriever_model(self):
        async with JAC(path=":memory:", retrieval="embeddings", embedder=FixtureEmbedder(),
                       jev=FixtureDecider()) as cache:
            call = {"args": [], "kwargs": _request("What is the capital of France?")}
            context = cache._context((), call["kwargs"])
            before = cache._cohort("function", call, context, None)
            cache.embedder.cache_identity = {"model": "changed"}
            self.assertNotEqual(before, cache._cohort("function", call, context, None))
            self.assertNotEqual(fingerprint(cache._retrieval_identity()), before)
