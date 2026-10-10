"""CUDA protocol tests with injected runtimes; no model download or GPU required."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from jac import JAC, Candidate, DecisionPolicy, JevK5CudaDecider
from jevcache.decisions import DecisionError
from jevcache.evaluate import providers


class Runtime:
    device = "cuda"
    temperature = 1.22

    def __init__(self, values=None, count=3):
        self.values = values if values is not None else {"true": .999, "false": .001}
        self.count = count
        self.calls = []

    def encode(self, state, instruction, options):
        return [1, 2, 3]

    def probabilities(self, state, question):
        self.calls.append((state, question))
        return self.values, self.count


class CudaTests(unittest.IsolatedAsyncioTestCase):
    def model(self, runtime=None, **kwargs):
        model = JevK5CudaDecider(runtime=runtime or Runtime(), **kwargs)
        model._options = lambda question: list(question["criteria"].items())
        self.addCleanup(model.close)
        return model

    async def decide(self, model):
        return await model.decide({"messages": [{"role": "user", "content": "new"}]}, [
            Candidate("candidate", {"text": "old"}, {"answer": "stored"}, 5000)
        ])

    async def test_complete_evidence_two_checks_and_no_generation(self):
        runtime = Runtime()
        model = self.model(runtime)
        result = await self.decide(model)
        self.assertEqual(result.candidate_id, "candidate")
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.calls[0][0]["artifact"], {"answer": "stored"})
        self.assertEqual(runtime.calls[0][0]["new_request"]["messages"][-1]["content"], "new")
        self.assertEqual(model.stats()["input_tokens"], 6)
        self.assertEqual(model.stats()["readout_tokens"], 0)

    async def test_token_budget_prevents_inference(self):
        runtime = Runtime()
        result = await self.decide(self.model(runtime, max_len=2))
        self.assertIsNone(result.candidate_id)
        self.assertEqual(runtime.calls, [])

    async def test_framework_reuses_semantic_response_without_another_generation(self):
        runtime = Runtime()
        model = self.model(runtime)
        await model.preload()
        calls = []

        async def generate(*, messages):
            calls.append(messages)
            await asyncio.sleep(.02)
            return {"answer": "Stored explanation of BM25."}

        first = {"messages": [{"role": "user", "content": "Explain BM25."}]}
        paraphrase = {"messages": [{"role": "user", "content": "Describe BM25."}]}
        async with JAC(path=":memory:", jev=model, candidate_limit=1,
                       decision_policy=DecisionPolicy(min_samples=1, overhead_ms=0)) as cache:
            ask = cache.wrap(generate)
            original = await ask(**first)
            ask.observe_reuse(paraphrase, valid_hit=True, decision_ms=.01)
            self.assertEqual(await ask(**paraphrase), original)
            self.assertEqual(cache.last_trace.path, "full_hit")
            self.assertEqual(cache.last_trace.jev_calls, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(runtime.calls), 2)

    async def test_missing_invalid_and_incomplete_probabilities_fail_closed(self):
        for values in ({"true": .99}, {"true": True, "false": 0},
                       {"true": float("nan"), "false": .1}, {"true": .9, "false": .9}):
            with self.subTest(values=values), self.assertRaises(DecisionError):
                await self.decide(self.model(Runtime(values)))
        with self.assertRaises(DecisionError):
            await self.decide(self.model(Runtime(count=2)))

    async def test_rejection_and_negative_question_polarity(self):
        self.assertIsNone((await self.decide(self.model(Runtime({"true": .1, "false": .9}))))
                          .candidate_id)
        model = self.model(Runtime({"true": .001, "false": .999}),
                           prompt_version="laya-reuse-v3-conflicts")
        self.assertEqual((await self.decide(model)).candidate_id, "candidate")

    async def test_no_silent_device_or_temperature_changes(self):
        runtime = Runtime()
        runtime.device = "cpu"
        with self.assertRaises(DecisionError):
            await self.decide(self.model(runtime))
        runtime.device, runtime.temperature = "cuda", 1
        with self.assertRaises(DecisionError):
            await self.decide(self.model(runtime))

    async def test_cancellation_prevents_remaining_checks(self):
        runtime = Runtime()
        entered, release = threading.Event(), threading.Event()
        original = runtime.probabilities

        def blocked(*args):
            entered.set()
            release.wait(2)
            return original(*args)

        runtime.probabilities = blocked
        model = self.model(runtime)
        task = asyncio.create_task(self.decide(model))
        await asyncio.to_thread(entered.wait, 1)
        self.assertTrue(entered.is_set())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()
        await model.drain()
        self.assertEqual(len(runtime.calls), 1)

    def test_identity_and_provider_configuration(self):
        model = self.model(revision="pinned")
        self.assertEqual(model.cache_identity["provider"], "jevk5-cuda")
        self.assertNotIn("endpoint", model.cache_identity)
        self.assertNotEqual(model.cache_identity, self.model(graphs=False).cache_identity)
        settings = {"provider": "jevk5-cuda", "revision": "pinned", "graphs": True}
        with patch("jevcache.evaluate.CompatibleClient"), \
                patch("jevcache.jevk5_cuda.JevK5CudaDecider") as constructor:
            providers({"generation": {"model": "remote"}, "decision": settings})
        constructor.assert_called_once_with(revision="pinned", graphs=True)

    def test_cpu_cannot_request_cuda_graphs(self):
        with self.assertRaises(ValueError):
            JevK5CudaDecider(device="cpu")
