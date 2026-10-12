"""Slow custom encoders must not delay unrelated hits or change loop affinity."""

import asyncio
import threading
import time
import unittest

from jac import JAC, Decision, DecisionPolicy
from jevcache.store import Entry, SQLiteStore


class AcceptDecider:
    cache_identity = {"model": "test-accept-v1"}

    async def decide(self, request, candidates, kind):
        return Decision(candidates[0].id, {}, "test-accept", 0)


class EmbeddingConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_sync_embedding_does_not_block_an_unrelated_exact_hit(self):
        entered = threading.Event()
        release = threading.Event()
        generations = []

        class Encoder:
            cache_identity = {"model": "test-slow-v1"}

            def embed(self, texts):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test embedding was not released")
                return [[1, 0] for _ in texts]

        async def generate(prompt):
            generations.append(prompt)
            await asyncio.sleep(0.030)
            return {"answer": "Returns are accepted within the policy window."}

        cache = JAC(":memory:", retrieval="embeddings", embedder=Encoder(),
                    decider=AcceptDecider(), decision_policy=DecisionPolicy(1, 0))
        pending = None
        try:
            ask = cache.wrap(generate)
            saved = await ask("Explain support return policy")
            ask.observe_reuse({}, request_args=("Describe support return policy",),
                              valid_hit=True, decision_ms=0)
            pending = asyncio.create_task(ask("Describe support return policy"))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            started = time.perf_counter()
            self.assertEqual(await ask("Explain support return policy"), saved)
            self.assertLess(time.perf_counter() - started, 0.250)
            self.assertFalse(release.is_set())
            self.assertEqual(cache.last_trace.llm_calls, 0)
            self.assertEqual(len(generations), 1)
            release.set()
            self.assertEqual(await pending, saved)
            self.assertEqual(len(generations), 1)
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await cache.drain()
            cache.close()

    async def test_async_encoder_runs_on_application_loop_in_both_modes(self):
        loop = asyncio.get_running_loop()
        for mode in ("cache_first", "parallel"):
            with self.subTest(mode=mode):
                calls = []

                class Encoder:
                    cache_identity = {"model": "test-loop-v1"}

                    async def aembed(self, texts):
                        calls.append((asyncio.get_running_loop(), texts))
                        await asyncio.sleep(0)
                        return [[1, 0] for _ in texts]

                async def generate(prompt):
                    if prompt.startswith("Describe"):
                        await asyncio.Event().wait()  # A semantic hit must win this race.
                    await asyncio.sleep(0.100)
                    return {"answer": "Saved support-policy answer."}

                async with JAC(":memory:", mode=mode, retrieval="embeddings",
                               embedder=Encoder(), decider=AcceptDecider(),
                               decision_policy=DecisionPolicy(1, 0)) as cache:
                    ask = cache.wrap(generate)
                    saved = await ask("Explain support return policy")
                    await cache.drain()
                    await ask("Explain support return policy")
                    await cache.drain()
                    self.assertEqual(calls, [])
                    ask.observe_reuse({}, request_args=("Describe support return policy",),
                                      valid_hit=True, decision_ms=0)
                    async with asyncio.timeout(2):
                        self.assertEqual(await ask("Describe support return policy"), saved)
                    await cache.drain()
                    self.assertEqual(cache.last_trace.path, "full_hit", cache.last_trace.to_dict())
                    self.assertEqual(len(calls), 1)
                    self.assertIs(calls[0][0], loop)

    async def test_async_encoder_failure_preserves_normal_generation(self):
        class Encoder:
            cache_identity = {"model": "test-unavailable-v1"}

            async def aembed(self, texts):
                raise RuntimeError("embedding endpoint unavailable")

        generations = []

        async def generate(prompt):
            generations.append(prompt)
            await asyncio.sleep(0.010)
            return {"generation": len(generations)}

        async with JAC(":memory:", retrieval="embeddings", embedder=Encoder(),
                       decider=AcceptDecider(), decision_policy=DecisionPolicy(1, 0)) as cache:
            ask = cache.wrap(generate)
            await ask("Explain support return policy")
            ask.observe_reuse({}, request_args=("Describe support return policy",),
                              valid_hit=True, decision_ms=0)
            self.assertEqual(await ask("Describe support return policy"), {"generation": 2})
            self.assertEqual(cache.last_trace.decision_status, "failed")
            self.assertEqual(cache.last_trace.jev_calls, 0)

    async def test_parallel_drain_tracks_cancelled_sync_encoder_work(self):
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = threading.Event()

        class Encoder:
            cache_identity = {"model": "test-sync-adapter-v1"}
            is_async = False

            def embed(self, texts):
                loop.call_soon_threadsafe(entered.set)
                if not release.wait(2):
                    raise TimeoutError("test embedding was not released")
                return [[1, 0] for _ in texts]

            async def aembed(self, texts):
                raise AssertionError("sync encoder must use the tracked worker lane")

        async def generate(prompt):
            if prompt.startswith("Describe"):
                await entered.wait()
                return {"answer": "new answer"}
            await asyncio.sleep(0.030)
            return {"answer": "saved answer"}

        cache = JAC(":memory:", mode="parallel", retrieval="embeddings", embedder=Encoder(),
                    decider=AcceptDecider(), decision_policy=DecisionPolicy(1, 0))
        try:
            ask = cache.wrap(generate)
            await ask("Explain support return policy")
            await cache.drain()
            ask.observe_reuse({}, request_args=("Describe support return policy",),
                              valid_hit=True, decision_ms=0)
            async with asyncio.timeout(2):
                self.assertEqual(await ask("Describe support return policy"),
                                 {"answer": "new answer"})
            draining = asyncio.create_task(cache.drain())
            for _ in range(5):
                await asyncio.sleep(0)
            self.assertFalse(draining.done())
            release.set()
            await draining
        finally:
            release.set()
            await cache.drain()
            cache.close()

    async def test_expired_exact_entry_is_not_returned_after_async_validation(self):
        now = [100.0]
        validations = []
        generations = []

        async def validate(request, response):
            validations.append(response)
            if len(validations) == 2:
                await asyncio.sleep(0)
                now[0] = 106.0
            return True

        async def generate(prompt):
            generations.append(prompt)
            return {"generation": len(generations)}

        async with JAC(":memory:", ttl_s=5, validator=validate) as cache:
            cache.store.clock = lambda: now[0]
            ask = cache.wrap(generate)
            self.assertEqual(await ask("hello"), {"generation": 1})
            self.assertEqual(await ask("hello"), {"generation": 2})
            self.assertEqual(cache.last_trace.path, "miss")

    async def test_expired_semantic_entry_is_not_returned_after_async_validation(self):
        now = [100.0]
        validations = []
        generations = []

        async def validate(request, response):
            validations.append(response)
            if len(validations) == 2:
                await asyncio.sleep(0)
                now[0] = 106.0
            return True

        async def generate(prompt):
            generations.append(prompt)
            await asyncio.sleep(0.020)
            return {"generation": len(generations)}

        async with JAC(":memory:", ttl_s=5, validator=validate,
                       decider=AcceptDecider(), decision_policy=DecisionPolicy(1, 0)) as cache:
            cache.store.clock = lambda: now[0]
            ask = cache.wrap(generate)
            await ask("Explain support return policy")
            ask.observe_reuse({}, request_args=("Describe support return policy",),
                              valid_hit=True, decision_ms=0)
            self.assertEqual(await ask("Describe support return policy"), {"generation": 2})
            self.assertEqual(cache.last_trace.path, "miss")

    async def test_expired_coalesced_entry_is_not_returned_after_waiter_validation(self):
        now = [100.0]
        entered = asyncio.Event()
        release = asyncio.Event()
        generations = []
        validations = []

        async def generate(prompt):
            generations.append(prompt)
            entered.set()
            await release.wait()
            return {"generation": len(generations)}

        async def validate(request, response):
            validations.append(response)
            if len(validations) == 2:
                await asyncio.sleep(0)
                now[0] = 106.0
            return True

        async with JAC(":memory:", ttl_s=5, validator=validate) as cache:
            cache.store.clock = lambda: now[0]
            ask = cache.wrap(generate)
            owner = asyncio.create_task(ask("hello"))
            await entered.wait()
            waiter = asyncio.create_task(ask("hello"))
            for _ in range(5):
                await asyncio.sleep(0)
            release.set()
            first, second = await asyncio.gather(owner, waiter)
            self.assertEqual(first, {"generation": 1})
            self.assertEqual(second, {"generation": 2})
            self.assertEqual(len(generations), 2)


class EmbeddingSnapshotTests(unittest.TestCase):
    def test_invalidated_entry_during_inference_is_not_restored(self):
        with SQLiteStore(clock=lambda: 10) as store:
            store.put(Entry("saved", "llm_answer", "test", "partition", {},
                            {"text": "support policy"}, {}, "1", "auto", 0, 100, 200))
            search = store.prepare_embedding_candidates(
                kind="llm_answer", scope="test", task="partition", text="returns policy",
                encoder_identity={"model": "test-v1"},
            )
            store.invalidate(scope="test")
            self.assertEqual(store.finish_embedding_candidates(search, [[1, 0], [1, 0]]), [])
            count = store.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            self.assertEqual(count, 0)

    def test_decider_alias_preserves_existing_jev_and_rejects_ambiguous_configuration(self):
        decider = AcceptDecider()
        with JAC(":memory:", decider=decider) as cache:
            self.assertIs(cache.jev, decider)
        with JAC(":memory:", jev=decider) as cache:
            self.assertIs(cache.jev, decider)
        with self.assertRaises(ValueError):
            JAC(":memory:", jev=decider, decider=decider)


if __name__ == "__main__":
    unittest.main()
