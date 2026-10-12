"""Application-owned model callbacks keep safe JAC contracts without a wire schema."""

import asyncio
import threading
import unittest

import jac
import jevcache
from jac import JAC, CallableDecider, CallableEmbedder, Candidate, Decision, DecisionPolicy
from jevcache.decisions import DecisionError

IDENTITY = {"model": "custom-model", "version": "weights-and-prompt-1"}


class ArrayLike:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class CallableEmbeddingTests(unittest.TestCase):
    def test_public_exports(self):
        self.assertIs(jac.CallableEmbedder, jevcache.CallableEmbedder)
        self.assertIs(jac.CallableDecider, jevcache.CallableDecider)

    def test_invalid_constructors(self):
        for adapter in (CallableEmbedder, CallableDecider):
            with self.assertRaises(TypeError):
                adapter(None, identity=IDENTITY)
        for threshold in (0, -1, 1.1, True, "1", float("nan"), float("inf"), 10**1000):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                CallableDecider(lambda *_: None, identity=IDENTITY, threshold=threshold)

    def test_identity_is_explicit_nonsecret_and_immutable(self):
        for identity in ({}, {"model": "m"}, {"model": "m", "version": ""},
                         {**IDENTITY, "api_key": "private"},
                         {**IDENTITY, "config": {"Authorization": "private"}},
                         {**IDENTITY, "connections": [{"refresh_token": "private"}]},
                         {**IDENTITY, "temperature": float("nan")}):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                CallableEmbedder(lambda texts: [[1] for _ in texts], identity=identity)
        original = {**IDENTITY, "preprocessing": {"prefix": "query:"}}
        adapter = CallableEmbedder(lambda texts: [[1] for _ in texts], identity=original)
        original["version"] = "different"
        adapter.cache_identity["identity"]["preprocessing"]["prefix"] = "different"
        self.assertEqual(adapter.cache_identity["identity"]["version"], IDENTITY["version"])
        self.assertEqual(adapter.cache_identity["identity"]["preprocessing"]["prefix"], "query:")

    def test_batch_order_array_conversion_and_empty_batch(self):
        seen = []

        def function(texts):
            seen.append(texts)
            return ArrayLike([[len(text), 1] for text in texts])

        adapter = CallableEmbedder(function, identity=IDENTITY)
        self.assertFalse(adapter.is_async)
        self.assertEqual(adapter.embed(["abcd", "x"]), [[4.0, 1.0], [1.0, 1.0]])
        self.assertEqual(adapter.embed([]), [])
        self.assertEqual(seen, [["abcd", "x"]])
        self.assertEqual(adapter.calls, 1)

    def test_invalid_batches_and_vectors(self):
        adapter = CallableEmbedder(lambda _: [[1]], identity=IDENTITY)
        for batch in ("one", [None], [1], {"text": "one"}):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                adapter.embed(batch)
        malformed = ([1], [], [[0, 0]], [[True, 1]], [[float("nan"), 1]],
                     [[float("inf")]], [["1"]], [[1], [1]], [[10**1000]], [[1.7e308, 1.7e308]])
        for value in malformed:
            bad = CallableEmbedder(lambda texts, value=value: value, identity=IDENTITY)
            with self.subTest(value=value), self.assertRaises(ValueError):
                bad.embed(["query"])

    def test_dimension_changes_are_rejected(self):
        output = [[1, 2]]
        adapter = CallableEmbedder(lambda _: output, identity=IDENTITY)
        self.assertEqual(adapter.embed(["query"]), [[1.0, 2.0]])
        output[:] = [[1, 2, 3]]
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            adapter.embed(["query"])
        inconsistent = CallableEmbedder(lambda _: [[1], [1, 2]], identity=IDENTITY)
        with self.assertRaisesRegex(ValueError, "consistent"):
            inconsistent.embed(["one", "two"])

    def test_sync_facade_for_async_function(self):
        async def function(texts):
            return [[len(text), 1] for text in texts]

        adapter = CallableEmbedder(function, identity=IDENTITY)
        self.assertTrue(adapter.is_async)
        self.assertEqual(adapter.embed(["hello"]), [[5.0, 1.0]])


class CallableProviderAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.deciders = []

    async def asyncTearDown(self):
        for decider in self.deciders:
            await decider.drain()
            decider.close()

    def decider(self, function, **kwargs):
        adapter = CallableDecider(function, identity=IDENTITY, **kwargs)
        self.deciders.append(adapter)
        return adapter

    @staticmethod
    def candidates():
        return [Candidate("alpha", {"text": "Original request"}, {"answer": "Answer"}, 50),
                Candidate("beta", {"text": "Another request"}, {"answer": "Other"}, 50)]

    async def test_async_encoder_runs_on_application_loop(self):
        loop = asyncio.get_running_loop()
        seen = []

        async def function(texts):
            self.assertIs(asyncio.get_running_loop(), loop)
            seen.append(texts)
            await asyncio.sleep(0)
            return [[1, len(text)] for text in texts]

        adapter = CallableEmbedder(function, identity=IDENTITY)
        self.assertEqual(await adapter.aembed(["hello"]), [[1.0, 5.0]])
        self.assertEqual(await adapter.aembed([]), [])
        self.assertEqual(seen, [["hello"]])
        with self.assertRaisesRegex(RuntimeError, "aembed"):
            adapter.embed(["hello"])

    async def test_sync_encoder_does_not_block_application_thread(self):
        application_thread = threading.get_ident()

        def function(texts):
            self.assertNotEqual(threading.get_ident(), application_thread)
            return [[1] for _ in texts]

        adapter = CallableEmbedder(function, identity=IDENTITY)
        self.assertEqual(await adapter.aembed(["one", "two"]), [[1.0], [1.0]])

    async def test_sync_function_returning_awaitable(self):
        loop = asyncio.get_running_loop()

        async def result():
            self.assertIs(asyncio.get_running_loop(), loop)
            return [[1, 2]]

        encoder = CallableEmbedder(lambda texts: result(), identity=IDENTITY)
        self.assertEqual(await encoder.aembed(["text"]), [[1.0, 2.0]])

    async def test_decisions_require_both_judgments_and_rank_conservatively(self):
        adapter = self.decider(lambda request, candidates, kind: {
            "alpha": {"equivalent": 0.999, "satisfies": 0.98},
            "beta": {"equivalent": True, "satisfies": 0.999},
        })
        result = await adapter.decide({"text": "New request"}, self.candidates(), "answer")
        self.assertEqual(result.candidate_id, "beta")
        self.assertEqual(result.probabilities["alpha:satisfies"], 0.98)
        self.assertEqual(result.model, IDENTITY["model"])
        self.assertGreaterEqual(result.elapsed_ms, 0)
        self.assertEqual(adapter.calls, 1)

    async def test_rejection_and_stable_tie_breaking(self):
        adapter = self.decider(lambda request, candidates, kind: {
            c.id: {"equivalent": True, "satisfies": False} for c in candidates
        })
        self.assertIsNone((await adapter.decide({}, self.candidates(), "answer")).candidate_id)
        accept = self.decider(lambda request, candidates, kind: {
            c.id: {"equivalent": True, "satisfies": True} for c in candidates
        })
        decision = await accept.decide({}, list(reversed(self.candidates())), "answer")
        self.assertEqual(decision.candidate_id, "alpha")
        self.assertNotEqual(accept.cache_identity, self.decider(
            lambda *_: None, threshold=0.9).cache_identity)

    async def test_async_decider_and_detached_inputs(self):
        loop = asyncio.get_running_loop()
        original = {"text": "New request"}
        candidates = self.candidates()

        async def function(request, options, kind):
            self.assertIs(asyncio.get_running_loop(), loop)
            self.assertEqual(kind, "intermediate")
            request["text"] = "mutated"
            options[0].request["text"] = "mutated"
            options[0].artifact["answer"] = "mutated"
            return {c.id: {"equivalent": False, "satisfies": False} for c in options}

        result = await self.decider(function).decide(original, candidates, "intermediate")
        self.assertIsNone(result.candidate_id)
        self.assertEqual(original["text"], "New request")
        self.assertEqual(candidates[0].request["text"], "Original request")
        self.assertEqual(candidates[0].artifact["answer"], "Answer")

    async def test_invalid_judgment_schema_and_unknown_ids_fail_closed(self):
        good = {"equivalent": True, "satisfies": True}
        malformed = (None, True, "alpha", {"alpha": good}, {"alpha": good, "foreign": good},
                     {"alpha": {"equivalent": True}, "beta": good},
                     {"alpha": {**good, "extra": True}, "beta": good})
        for output in malformed:
            adapter = self.decider(lambda request, candidates, kind, value=output: value)
            with self.subTest(output=output), self.assertRaises(DecisionError):
                await adapter.decide({}, self.candidates(), "answer")
        for bad in (float("nan"), float("inf"), -0.1, 1.1, "yes", None, [], 10**1000):
            adapter = self.decider(lambda request, candidates, kind, value=bad: {
                "alpha": {"equivalent": value, "satisfies": True}, "beta": good,
            })
            with self.subTest(bad=bad), self.assertRaises(DecisionError):
                await adapter.decide({}, self.candidates(), "answer")

    async def test_native_decision_callback_is_validated(self):
        valid = self.decider(lambda *_: Decision("alpha", {}, IDENTITY["model"], 123))
        result = await valid.decide({}, self.candidates(), "plan")
        self.assertEqual(result.candidate_id, "alpha")
        self.assertNotEqual(result.elapsed_ms, 123)
        for result in (Decision("foreign", {}, IDENTITY["model"], 0),
                       Decision(["alpha"], {}, IDENTITY["model"], 0),
                       Decision("alpha", {}, "wrong-model", 0),
                       Decision("alpha", {"score": True}, IDENTITY["model"], 0),
                       Decision("alpha", {"score": float("nan")}, IDENTITY["model"], 0)):
            adapter = self.decider(lambda request, candidates, kind, result=result: result)
            with self.subTest(result=result), self.assertRaises(DecisionError):
                await adapter.decide({}, self.candidates(), "answer")

    async def test_empty_candidates_skip_callback_and_invalid_inputs(self):
        adapter = self.decider(lambda *_: self.fail("Empty candidates must not invoke model"))
        self.assertIsNone((await adapter.decide({}, [], "answer")).candidate_id)
        self.assertEqual(adapter.calls, 0)
        for candidates in ([self.candidates()[0]] * 2, self.candidates() * 5,
                           [Candidate("", {}, "answer", 50)],
                           [Candidate("alpha", {}, "answer", float("nan"))],
                           [Candidate("alpha", {}, "answer", 10**1000)],
                           [Candidate("alpha", {}, "answer", True)]):
            with self.subTest(candidates=candidates), self.assertRaises(DecisionError):
                await adapter.decide({}, candidates, "answer")
        with self.assertRaises(DecisionError):
            await adapter.decide({}, self.candidates(), "arbitrary-generation")
        with self.assertRaises(DecisionError):
            await adapter.decide({"bad": float("nan")}, self.candidates(), "answer")

    async def test_provider_exception_details_are_not_propagated(self):
        def failed(*_):
            raise ValueError("api-key-private-and-user-data")

        with self.assertRaisesRegex(DecisionError, "^Decision function failed$"):
            await self.decider(failed).decide({}, self.candidates(), "answer")
        with self.assertRaisesRegex(ValueError, "^Embedding function failed$"):
            await CallableEmbedder(failed, identity=IDENTITY).aembed(["text"])

    async def test_timed_out_sync_decision_does_not_queue_more_work(self):
        entered, release = threading.Event(), threading.Event()

        def function(request, candidates, kind):
            entered.set()
            release.wait(timeout=5)
            return {c.id: {"equivalent": True, "satisfies": True} for c in candidates}

        adapter = self.decider(function)
        task = asyncio.create_task(adapter.decide({}, self.candidates(), "answer"))
        await asyncio.to_thread(entered.wait, 2)
        self.assertTrue(entered.is_set())
        task.cancel()
        try:
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaisesRegex(DecisionError, "still running"):
                await adapter.decide({}, self.candidates(), "answer")
            self.assertEqual(adapter.calls, 1)
        finally:
            release.set()
            await adapter.drain()
        self.assertEqual((await adapter.decide({}, self.candidates(), "answer")).candidate_id,
                         "alpha")
        adapter.close()
        with self.assertRaisesRegex(DecisionError, "closed"):
            await adapter.decide({}, self.candidates(), "answer")

    async def test_custom_models_in_complete_jac_semantic_reuse_path(self):
        application_loop = asyncio.get_running_loop()
        generation_calls = []

        async def generate(*, prompt):
            generation_calls.append(prompt)
            await asyncio.sleep(0.03)
            return {"answer": "Use exponential backoff with bounded retries."}

        async def embed(texts):
            self.assertIs(asyncio.get_running_loop(), application_loop)
            return [[1, 1] for _ in texts]

        async def judge(request, candidates, kind):
            self.assertIs(asyncio.get_running_loop(), application_loop)
            return {c.id: {"equivalent": True, "satisfies": True} for c in candidates}

        encoder, decider = CallableEmbedder(embed, identity=IDENTITY), self.decider(judge)
        async with JAC(path=":memory:", jev=decider, retrieval="embeddings", embedder=encoder,
                       decision_policy=DecisionPolicy(1, 0)) as cache:
            wrapped = cache.wrap(generate, namespace="support-agent-v1")
            first = await wrapped(prompt="How should our API client retry temporary failures?")
            second_request = {"prompt": "Suggest an API retry policy for transient errors."}
            wrapped.observe_reuse(second_request, valid_hit=True, decision_ms=0)
            second = await wrapped(**second_request)
            self.assertEqual(second, first)
            self.assertEqual(cache.last_trace.path, "full_hit")
            self.assertEqual(cache.last_trace.llm_calls, 0)
            self.assertEqual(len(generation_calls), 1)
            self.assertEqual(encoder.calls, 1)
            self.assertEqual(decider.calls, 1)
            await wrapped(**second_request)
            self.assertEqual(encoder.calls, 1)
            self.assertEqual(decider.calls, 1)

    async def test_invalid_custom_decision_falls_back_to_normal_generation(self):
        generated = []

        async def generate(*, prompt):
            generated.append(prompt)
            await asyncio.sleep(0.03)
            return {"answer": prompt}

        decider = self.decider(lambda *_: {"not-an-eligible-id": {
            "equivalent": True, "satisfies": True,
        }})
        async with JAC(path=":memory:", jev=decider,
                       decision_policy=DecisionPolicy(1, 0)) as cache:
            wrapped = cache.wrap(generate)
            await wrapped(prompt="Support API retry failure")
            request = {"prompt": "Explain API retry failure"}
            wrapped.observe_reuse(request, valid_hit=True, decision_ms=0)
            response = await wrapped(**request)
            self.assertEqual(response, {"answer": request["prompt"]})
            self.assertEqual(len(generated), 2)
            self.assertEqual(cache.last_trace.decision_status, "failed")


if __name__ == "__main__":
    unittest.main()
