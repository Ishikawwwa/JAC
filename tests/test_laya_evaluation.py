"""Provider wiring and local-generation cancellation, without model downloads."""

import asyncio
import json
import threading
import unittest
from unittest.mock import patch

from jevcache import evaluate
from jevcache.local_generation import TransformersClient


class FastClient:
    calls = 0
    completed = 0
    output_tokens = 0
    cache_identity = {"provider": "injected-test-generator"}

    async def complete(self, **kwargs):
        self.calls += 1
        self.completed += 1
        self.output_tokens += 1
        text = kwargs["messages"][-1]["content"]
        content = json.dumps({"answer": evaluate.ORACLE[text]})
        return {"choices": [{"message": {"content": content}}]}

    async def drain(self):
        pass


class PreloadedFixture(evaluate.FixtureDecider):
    def __init__(self):
        self.preloads = self.drains = 0

    async def preload(self):
        self.preloads += 1

    async def drain(self):
        self.drains += 1

    @property
    def cache_identity(self):
        return {"provider": "injected-test-decider", "preloaded": self.preloads > 0}


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def test_native_provider_forwards_supported_pinned_settings(self):
        generation_settings = {"provider": "transformers", "model": "local-generator",
                               "revision": "generator-sha", "device": "cpu", "max_new_tokens": 32}
        decision_settings = {"provider": "laya", "model": "convaiinnovations/laya",
                             "revision": "decision-sha", "device": "cuda", "dtype": "float16",
                             "threshold": .9, "max_len": 512,
                             "prompt_version": "laya-reuse-v3-bindings", "unused": "not-passed"}
        with patch("jevcache.local_generation.TransformersClient") as generator_constructor:
            with patch("jevcache.laya.LayaDecider") as decider_constructor:
                generator, decider = evaluate.providers({"generation": generation_settings,
                                                        "decision": decision_settings})
        generator_constructor.assert_called_once_with(model="local-generator",
                                                       revision="generator-sha", device="cpu",
                                                       max_new_tokens=32)
        decider_constructor.assert_called_once_with(model="convaiinnovations/laya",
                                                     revision="decision-sha", device="cuda",
                                                     dtype="float16", threshold=.9, max_len=512,
                                                     prompt_version="laya-reuse-v3-bindings")
        self.assertIs(generator, generator_constructor.return_value)
        self.assertIs(decider, decider_constructor.return_value)

    async def test_evaluation_preloads_and_drains_native_decider_outside_trials(self):
        generator, decider = FastClient(), PreloadedFixture()
        with patch("jevcache.evaluate.providers", return_value=(generator, decider)):
            report = await evaluate.run_evaluation(
                config={}, offline=False, retrieval="bm25", mode="cache_first", repeats=1
            )
        self.assertEqual(decider.preloads, 1)
        self.assertTrue(report["decision_provider"]["preloaded"])
        self.assertEqual(set(report["runs"]), {"no_cache", "exact_only", "bm25_cache_first"})
        baseline = report["runs"]["no_cache"]
        self.assertEqual(baseline["generation_calls"], baseline["requests"])
        self.assertEqual(baseline["generation_completions"], baseline["generation_calls"])
        self.assertEqual(baseline["generated_tokens"], baseline["generation_calls"])
        self.assertEqual(baseline["incorrect_responses"], 0)
        self.assertEqual(decider.drains, sum(row["requests"] for row in report["runs"].values()))


class LocalGenerationTests(unittest.IsolatedAsyncioTestCase):
    def client(self, generate):
        # Bypass optional model construction and exercise the real async bridge.
        client = TransformersClient.__new__(TransformersClient)
        client.calls = client.completed = client.output_tokens = 0
        client._workers = set()
        client._generate = generate
        return client

    async def test_caller_cancellation_preserves_started_work_until_drain(self):
        entered, release = threading.Event(), threading.Event()

        def generate(kwargs):
            entered.set()
            release.wait(2)
            client.completed += 1
            return {"answer": "finished"}

        client = self.client(generate)
        first = asyncio.create_task(client.complete(messages=[]))
        await asyncio.to_thread(entered.wait, 1)
        self.assertTrue(entered.is_set())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.completed, 0)
        self.assertEqual(len(client._workers), 1)
        release.set()
        await client.drain()
        self.assertEqual(client.completed, 1)

    async def test_cancelled_drain_does_not_drop_a_running_thread(self):
        entered, release = threading.Event(), threading.Event()

        def generate(kwargs):
            entered.set()
            release.wait(2)
            client.completed += 1
            return {"answer": "finished"}

        client = self.client(generate)
        caller = asyncio.create_task(client.complete(messages=[]))
        await asyncio.to_thread(entered.wait, 1)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        drain = asyncio.create_task(client.drain())
        await asyncio.sleep(0)
        drain.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await drain
        self.assertEqual(len(client._workers), 1)
        release.set()
        await client.drain()
        self.assertEqual(client.completed, 1)

    async def test_abandoned_worker_error_is_observed_without_loop_error_log(self):
        entered, release = threading.Event(), threading.Event()

        def generate(kwargs):
            entered.set()
            release.wait(2)
            raise RuntimeError("intentional test error")

        client = self.client(generate)
        errors = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda event_loop, context: errors.append(context))
        try:
            caller = asyncio.create_task(client.complete(messages=[]))
            await asyncio.to_thread(entered.wait, 1)
            caller.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await caller
            release.set()
            await asyncio.sleep(.02)
            await asyncio.sleep(0)
            self.assertEqual(errors, [])
        finally:
            loop.set_exception_handler(previous)


if __name__ == "__main__":
    unittest.main()
