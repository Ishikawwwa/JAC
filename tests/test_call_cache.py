"""The primary JAC contract: skip generation while preserving the call's response."""

import asyncio
import copy
import inspect
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from jevcache.call_cache import JAC
from jevcache.decisions import Decision, DecisionPolicy, JevDecider
from jevcache.models import Context


class FakeResponse:
    def __init__(self, data):
        self.data = copy.deepcopy(data)

    def model_dump(self, **_kwargs):
        return copy.deepcopy(self.data)

    @classmethod
    def model_validate(cls, data):
        return cls(data)


class FakeDecider:
    model = "fake-jev-fixture"

    def __init__(self, *, accept=True, malformed=False, foreign=False, error=None):
        self.accept = accept
        self.malformed = malformed
        self.foreign = foreign
        self.error = error
        self.calls = []

    async def decide(self, request, candidates, kind):
        self.calls.append((request, candidates, kind))
        if self.error:
            raise self.error
        if self.malformed:
            return {"candidate_id": candidates[0].id}
        candidate_id = ("foreign-entry" if self.foreign else candidates[0].id) \
            if self.accept else None
        return Decision(candidate_id, {}, self.model, 0.0)


class _CacheTests:
    def cache(self, **kwargs):
        jac = JAC(path=":memory:", **kwargs)
        self.caches.append(jac)
        return jac

    def close_caches(self):
        for jac in self.caches:
            close = getattr(jac, "close", None)
            if close is not None:
                close()
            else:
                jac.store.close()


class SyncCallCacheTests(_CacheTests, unittest.TestCase):
    def setUp(self):
        self.caches = []

    def tearDown(self):
        self.close_caches()

    def test_exact_hit_returns_same_json_response_without_generating(self):
        generated = []

        def generate(prompt, *, model="fixture", temperature=0):
            generated.append((prompt, model, temperature))
            return {"content": "A saved response", "usage": {"tokens": 7}}

        jac = self.cache()
        cached = jac.wrap(generate)
        self.assertEqual(inspect.signature(cached), inspect.signature(generate))
        first = cached("Hello", model="fixture", temperature=0)
        second = cached("Hello", model="fixture", temperature=0)
        self.assertEqual(first, second)
        self.assertEqual(len(generated), 1)
        self.assertEqual(jac.last_trace.llm_calls, 0)
        self.assertEqual(jac.last_trace.jev_calls, 0)

    def test_cache_miss_executes_original_function_with_unchanged_arguments(self):
        calls = []

        def generate(prompt, /, model="fixture", *, temperature=0):
            calls.append((prompt, model, temperature))
            return f"{prompt}:{model}:{temperature}"

        jac = self.cache()
        cached = jac.wrap(generate)
        self.assertEqual(cached("one", "model-a", temperature=0.2), "one:model-a:0.2")
        self.assertEqual(cached("two", "model-a", temperature=0.2), "two:model-a:0.2")
        self.assertEqual(calls, [("one", "model-a", 0.2), ("two", "model-a", 0.2)])
        self.assertEqual(jac.last_trace.llm_calls, 1)

    def test_model_generation_settings_and_tools_are_part_of_exact_key(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return {"call": len(calls)}

        jac = self.cache()
        cached = jac.wrap(generate)
        baseline = {"prompt": "Hello", "model": "a", "temperature": 0,
                    "max_tokens": 100, "tools": [{"name": "read"}]}
        original = cached(**baseline)
        changes = ({"model": "b"}, {"temperature": 0.5}, {"max_tokens": 200},
                   {"tools": [{"name": "search"}]})
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(cached(**(baseline | change)), original)
        self.assertEqual(cached(**baseline), original)
        self.assertEqual(len(calls), 1 + len(changes))

    def test_system_prompt_and_prior_conversation_history_are_part_of_key(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return {"call": len(calls)}

        cached = self.cache().wrap(generate)
        messages = [{"role": "system", "content": "Reply briefly"},
                    {"role": "assistant", "content": "Earlier answer"},
                    {"role": "user", "content": "Explain returns"}]
        baseline = cached(model="fixture", messages=messages)
        system_changed = copy.deepcopy(messages)
        system_changed[0]["content"] = "Reply in German"
        history_changed = copy.deepcopy(messages)
        history_changed[1]["content"] = "A different earlier answer"
        self.assertNotEqual(cached(model="fixture", messages=system_changed), baseline)
        self.assertNotEqual(cached(model="fixture", messages=history_changed), baseline)
        self.assertEqual(cached(model="fixture", messages=messages), baseline)
        self.assertEqual(len(calls), 3)

    def test_scope_dependency_and_policy_changes_invalidate_response(self):
        calls = []
        current = {"scope": "tenant-a", "revision": "1", "policy": "1"}

        def generate(prompt):
            calls.append(prompt)
            return {"generation": len(calls)}

        jac = self.cache(context=lambda args, kwargs: Context(
            current["scope"], {"document": current["revision"]}, current["policy"]
        ))
        cached = jac.wrap(generate)
        first = cached("Hello")
        self.assertEqual(cached("Hello"), first)
        current["scope"] = "tenant-b"
        self.assertNotEqual(cached("Hello"), first)
        current["revision"] = "2"
        self.assertEqual(cached("Hello")["generation"], 3)
        current["policy"] = "2"
        self.assertEqual(cached("Hello")["generation"], 4)
        self.assertEqual(len(calls), 4)

    def test_function_namespaces_partition_results(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return {"generation": len(calls)}

        jac = self.cache()
        first = jac.wrap(generate, namespace="agent-a")
        second = jac.wrap(generate, namespace="agent-b")
        self.assertEqual(first("Hello")["generation"], 1)
        self.assertEqual(second("Hello")["generation"], 2)
        self.assertEqual(first("Hello")["generation"], 1)
        self.assertEqual(second("Hello")["generation"], 2)
        self.assertEqual(len(calls), 2)

    def test_ttl_expiry_forces_new_generation(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return {"generation": len(calls)}

        jac = self.cache(ttl_s=5)
        now = [1000.0]
        jac.store.clock = lambda: now[0]
        cached = jac.wrap(generate)
        self.assertEqual(cached("Hello")["generation"], 1)
        now[0] += 4
        self.assertEqual(cached("Hello")["generation"], 1)
        now[0] += 1
        self.assertEqual(cached("Hello")["generation"], 2)

    def test_zero_ttl_does_not_cache(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return "answer"

        cached = self.cache(ttl_s=0).wrap(generate)
        cached("Hello")
        cached("Hello")
        self.assertEqual(len(calls), 2)

    def test_provider_exception_is_not_cached_or_changed(self):
        error = ValueError("provider failure")
        count = 0

        def generate(prompt):
            nonlocal count
            count += 1
            raise error

        cached = self.cache().wrap(generate)
        for _ in range(2):
            with self.assertRaises(ValueError) as caught:
                cached("Hello")
            self.assertIs(caught.exception, error)
        self.assertEqual(count, 2)

    def test_returned_json_mutation_cannot_change_future_hits(self):
        original = {"content": [{"text": "Original answer"}]}
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return original

        cached = self.cache().wrap(generate)
        first = cached("Hello")
        first["content"][0]["text"] = "Caller mutation"
        second = cached("Hello")
        self.assertEqual(second["content"][0]["text"], "Original answer")
        second["content"].append({"text": "Another mutation"})
        self.assertEqual(cached("Hello"), {"content": [{"text": "Original answer"}]})
        self.assertEqual(len(calls), 1)

    def test_validator_can_prevent_admission_and_reject_existing_hit(self):
        calls = []
        allowed = [False]

        def generate(prompt):
            calls.append(prompt)
            return {"generation": len(calls)}

        def validate(request, response):
            self.assertEqual(request["args"], ["Hello"])
            self.assertIsInstance(response, dict)
            return allowed[0]

        jac = self.cache(validator=validate)
        cached = jac.wrap(generate)
        self.assertEqual(cached("Hello")["generation"], 1)
        self.assertEqual(cached("Hello")["generation"], 2)
        allowed[0] = True
        self.assertEqual(cached("Hello")["generation"], 3)
        self.assertEqual(cached("Hello")["generation"], 3)
        allowed[0] = False
        self.assertEqual(cached("Hello")["generation"], 4)
        self.assertEqual(len(calls), 4)

    def test_cache_io_failure_does_not_replace_original_function_result(self):
        for method in ("get", "put", "record_trace"):
            with self.subTest(method=method):
                jac = self.cache()
                calls = []

                def generate(prompt):
                    calls.append(prompt)
                    return {"answer": "original response"}

                cached = jac.wrap(generate)
                with patch.object(jac.store, method, side_effect=OSError("cache unavailable")):
                    self.assertEqual(cached("Hello"), {"answer": "original response"})
                self.assertEqual(len(calls), 1)

    def test_stats_count_generation_and_exact_hits(self):
        jac = self.cache()
        cached = jac.wrap(lambda prompt: {"answer": prompt})
        cached("one")
        cached("one")
        cached("two")
        stats = jac.stats()
        self.assertEqual(stats["requests"], 3)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["llm_calls"], 2)
        self.assertEqual(stats["jev_calls"], 0)

    def test_sync_semantic_reuse_skips_generation(self):
        decider = FakeDecider()
        jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=0
        ))
        calls = []

        def generate(prompt):
            calls.append(prompt)
            time.sleep(0.030)
            return {"answer": "A reusable answer"}

        cached = jac.wrap(generate)
        first = cached("Explain Orchid return policy")
        for _ in range(20):
            cached.observe_reuse({}, request_args=("Describe Orchid return policy",),
                                 valid_hit=True, decision_ms=0)
        self.assertEqual(cached("Describe Orchid return policy"), first)
        self.assertEqual(len(calls), 1)
        self.assertEqual(jac.last_trace.jev_calls, 1)
        self.assertEqual(jac.last_trace.llm_calls, 0)

    def test_typed_response_codec_preserves_type_and_independent_instances(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return FakeResponse({"choices": [{"text": "answer"}]})

        cached = self.cache().wrap(generate, response_type=FakeResponse)
        first = cached("Hello")
        second = cached("Hello")
        self.assertIsInstance(first, FakeResponse)
        self.assertIsInstance(second, FakeResponse)
        self.assertIsNot(first, second)
        second.data["choices"][0]["text"] = "mutated"
        self.assertEqual(cached("Hello").model_dump(), {"choices": [{"text": "answer"}]})
        self.assertEqual(len(calls), 1)

    def test_persistent_typed_hit_with_explicit_response_type_never_generates(self):
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return FakeResponse({"answer": "persistent"})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite"
            first = JAC(path=path)
            second = None
            try:
                first.wrap(generate, response_type=FakeResponse)("Hello")
                first.close()
                second = JAC(path=path)
                result = second.wrap(generate, response_type=FakeResponse)("Hello")
                self.assertIsInstance(result, FakeResponse)
                self.assertEqual(result.model_dump(), {"answer": "persistent"})
                self.assertEqual(len(calls), 1)
            finally:
                if second is not None:
                    second.close()

    def test_unsupported_request_is_forwarded_without_cache(self):
        opaque = object()
        calls = []

        def generate(prompt, attachment):
            self.assertIs(attachment, opaque)
            calls.append(prompt)
            return "answer"

        cached = self.cache().wrap(generate)
        self.assertEqual(cached("Hello", opaque), "answer")
        self.assertEqual(cached("Hello", opaque), "answer")
        self.assertEqual(len(calls), 2)

    def test_unsupported_response_is_returned_unchanged_without_cache(self):
        response = object()
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return response

        cached = self.cache().wrap(generate)
        self.assertIs(cached("Hello"), response)
        self.assertIs(cached("Hello"), response)
        self.assertEqual(len(calls), 2)

    def test_tuple_response_bypasses_generic_json_cache_without_changing_type(self):
        response = ("answer", {"usage": 7})
        calls = []

        def generate(prompt):
            calls.append(prompt)
            return response

        cached = self.cache().wrap(generate)
        self.assertIs(cached("Hello"), response)
        self.assertIs(cached("Hello"), response)
        self.assertEqual(len(calls), 2)

    def test_streaming_calls_return_original_stream_and_always_generate(self):
        stream = iter(["one", "two"])
        calls = []

        def generate(prompt, stream=False):
            calls.append((prompt, stream))
            return stream_object

        stream_object = stream
        cached = self.cache().wrap(generate)
        self.assertIs(cached("Hello", stream=True), stream_object)
        self.assertIs(cached("Hello", stream=True), stream_object)
        self.assertEqual(len(calls), 2)


class AsyncCallCacheTests(_CacheTests, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.caches = []

    def tearDown(self):
        self.close_caches()

    async def test_async_exact_hit_returns_response_and_skips_generation(self):
        calls = []

        async def generate(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return {"answer": "cached"}

        jac = self.cache()
        cached = jac.wrap(generate)
        self.assertTrue(inspect.iscoroutinefunction(cached))
        self.assertEqual(await cached("Hello", model="fixture"), {"answer": "cached"})
        self.assertEqual(await cached("Hello", model="fixture"), {"answer": "cached"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(jac.last_trace.llm_calls, 0)

    async def test_simultaneous_identical_async_requests_generate_once(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            entered.set()
            await release.wait()
            return {"answer": ["coalesced"]}

        cached = self.cache().wrap(generate)
        first = asyncio.create_task(cached("Hello"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        second = asyncio.create_task(cached("Hello"))
        for _ in range(10):
            await asyncio.sleep(0)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        self.assertEqual(calls, ["Hello"])
        self.assertEqual(results, [{"answer": ["coalesced"]}] * 2)
        self.assertIsNot(results[0], results[1])
        results[0]["answer"].append("caller mutation")
        self.assertEqual(results[1], {"answer": ["coalesced"]})

    async def test_async_exception_is_not_cached(self):
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            raise RuntimeError("provider unavailable")

        cached = self.cache().wrap(generate)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                await cached("Hello")
        self.assertEqual(len(calls), 2)

    async def test_async_validator_is_supported(self):
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            return {"answer": prompt}

        async def validate(request, response):
            await asyncio.sleep(0)
            return request["args"][0] == response["answer"]

        cached = self.cache(validator=validate).wrap(generate)
        self.assertEqual(await cached("Hello"), {"answer": "Hello"})
        self.assertEqual(await cached("Hello"), {"answer": "Hello"})
        self.assertEqual(len(calls), 1)

    async def test_async_stream_response_passes_through_unchanged(self):
        async def events():
            yield "answer"

        stream = events()
        calls = []

        async def generate(prompt, **kwargs):
            calls.append((prompt, kwargs))
            return stream

        cached = self.cache().wrap(generate)
        try:
            self.assertIs(await cached("Hello", stream=True), stream)
            self.assertIs(await cached("Hello", stream=True), stream)
            self.assertEqual(len(calls), 2)
        finally:
            await stream.aclose()

    async def semantic_cache(self, decider=None, *, overhead_ms=0):
        decider = decider or FakeDecider()
        jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=overhead_ms
        ))
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.030)
            return {"answer": "Returns accepted subject to the current policy."}

        cached = jac.wrap(generate)
        await cached(prompt="Explain Orchid return policy", model="fixture")
        return jac, cached, decider, calls

    @staticmethod
    def observe(cached, kwargs, *, valid_hit=True, decision_ms=0):
        for _ in range(20):
            cached.observe_reuse(kwargs, valid_hit=valid_hit, decision_ms=decision_ms)

    async def test_semantic_cold_start_executes_without_decision(self):
        jac, cached, decider, calls = await self.semantic_cache()
        await cached(prompt="Describe Orchid return policy", model="fixture")
        self.assertEqual(len(calls), 2)
        self.assertEqual(decider.calls, [])
        self.assertEqual(jac.last_trace.jev_calls, 0)

    async def test_semantic_hit_uses_one_decision_and_no_generation_then_becomes_exact(self):
        jac, cached, decider, calls = await self.semantic_cache()
        kwargs = {"prompt": "Describe Orchid return policy", "model": "fixture"}
        self.observe(cached, kwargs)
        result = await cached(**kwargs)
        self.assertEqual(result, {"answer": "Returns accepted subject to the current policy."})
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(decider.calls), 1)
        self.assertEqual(jac.last_trace.llm_calls, 0)
        self.assertEqual(jac.last_trace.jev_calls, 1)
        self.assertEqual(decider.calls[0][2], "answer")
        self.assertEqual(await cached(**kwargs), result)
        self.assertEqual(jac.last_trace.llm_calls, 0)
        self.assertEqual(jac.last_trace.jev_calls, 0)
        self.assertEqual(len(decider.calls), 1)

    async def test_expensive_decision_or_zero_measured_hits_bypasses_jev(self):
        for overhead, valid in ((100_000, True), (0, False)):
            with self.subTest(overhead=overhead, valid=valid):
                jac, cached, decider, calls = await self.semantic_cache(overhead_ms=overhead)
                kwargs = {"prompt": "Describe Orchid return policy", "model": "fixture"}
                self.observe(cached, kwargs, valid_hit=valid)
                await cached(**kwargs)
                self.assertEqual(len(calls), 2)
                self.assertEqual(decider.calls, [])
                self.assertEqual(jac.last_trace.jev_calls, 0)

    async def test_rejected_or_malformed_decision_falls_back_and_records_attempt(self):
        for decider in (FakeDecider(accept=False), FakeDecider(malformed=True),
                        FakeDecider(foreign=True), FakeDecider(error=RuntimeError("offline"))):
            with self.subTest(decider=decider):
                jac, cached, _, calls = await self.semantic_cache(decider)
                kwargs = {"prompt": "Describe Orchid return policy", "model": "fixture"}
                self.observe(cached, kwargs)
                result = await cached(**kwargs)
                self.assertEqual(result["answer"],
                                 "Returns accepted subject to the current policy.")
                self.assertEqual(len(calls), 2)
                self.assertEqual(jac.last_trace.llm_calls, 1)
                self.assertEqual(jac.last_trace.jev_calls, 1)
                decision_events = [event for event in jac.last_trace.events
                                   if event.kind == "jev"]
                self.assertEqual(len(decision_events), 1)
                self.assertIn(decision_events[0].status, ("rejected", "failed"))

    async def test_timeout_falls_back_and_includes_elapsed_decision(self):
        calls = []

        async def transport(payload):
            calls.append(payload)
            await asyncio.sleep(0.5)
            return {}

        decider = JevDecider(transport=transport, timeout_s=0.001)
        jac, cached, _, generated = await self.semantic_cache(decider)
        kwargs = {"prompt": "Describe Orchid return policy", "model": "fixture"}
        self.observe(cached, kwargs)
        await cached(**kwargs)
        self.assertEqual(len(generated), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(jac.last_trace.jev_calls, 1)
        events = [event for event in jac.last_trace.events if event.kind == "jev"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].status, "failed")
        self.assertGreater(events[0].elapsed_ms, 0)

    async def test_changed_numeric_anchor_cannot_reach_decider(self):
        decider = FakeDecider()
        jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=0
        ))
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.030)
            return {"answer": kwargs["prompt"]}

        cached = jac.wrap(generate)
        await cached(prompt="Explain returns within 30 days", model="fixture")
        kwargs = {"prompt": "Explain returns within 90 days", "model": "fixture"}
        self.observe(cached, kwargs)
        result = await cached(**kwargs)
        self.assertEqual(result, {"answer": "Explain returns within 90 days"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(decider.calls, [])
        self.assertEqual(jac.last_trace.jev_calls, 0)

    async def test_numeric_sign_percentage_and_order_changes_block_semantic_reuse(self):
        pairs = [
            ("Compute leverage with +5% haircut", "Compute leverage with -5% haircut"),
            ("Compute leverage with 5% haircut", "Compute leverage with 5 haircut"),
            ("Transfer 10 from account 20", "Transfer 20 from account 10"),
        ]
        for old, new in pairs:
            with self.subTest(old=old, new=new):
                decider = FakeDecider()
                jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
                    min_samples=20, overhead_ms=0
                ))
                calls = []

                async def generate(**kwargs):
                    calls.append(kwargs)
                    await asyncio.sleep(0.030)
                    return {"answer": kwargs["prompt"]}

                cached = jac.wrap(generate)
                await cached(prompt=old)
                kwargs = {"prompt": new}
                self.observe(cached, kwargs)
                self.assertEqual(await cached(**kwargs), {"answer": new})
                self.assertEqual(len(calls), 2)
                self.assertEqual(decider.calls, [])
                self.assertEqual(jac.last_trace.jev_calls, 0)

    async def test_candidate_expiring_during_semantic_decision_is_not_returned(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        class DelayedDecider(FakeDecider):
            async def decide(self, request, candidates, kind):
                self.calls.append((request, candidates, kind))
                entered.set()
                await release.wait()
                return Decision(candidates[0].id, {}, self.model, 0)

        decider = DelayedDecider()
        now = [1000.0]
        jac = self.cache(ttl_s=5, jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=0
        ))
        jac.store.clock = lambda: now[0]
        generations = 0

        async def generate(**kwargs):
            nonlocal generations
            generations += 1
            await asyncio.sleep(0.030)
            return {"generation": generations}

        cached = jac.wrap(generate)
        await cached(prompt="Explain Orchid return policy")
        kwargs = {"prompt": "Describe Orchid return policy"}
        self.observe(cached, kwargs)

        async def call_and_trace():
            value = await cached(**kwargs)
            return value, jac.last_trace

        pending = asyncio.create_task(call_and_trace())
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            now[0] += 6
        finally:
            release.set()
        value, trace = await asyncio.wait_for(pending, timeout=1)
        self.assertEqual(value, {"generation": 2})
        self.assertEqual(generations, 2)
        self.assertEqual(trace.llm_calls, 1)
        self.assertEqual(trace.jev_calls, 1)

    async def test_changed_decider_identity_invalidates_promoted_exact_alias(self):
        class ConfigDecider(FakeDecider):
            def __init__(self, version, **kwargs):
                super().__init__(**kwargs)
                self.version = version

            @property
            def cache_identity(self):
                return {"configuration_version": self.version}

        decider = ConfigDecider("1")
        jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=0
        ))
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.030)
            return {"generation": len(calls)}

        cached = jac.wrap(generate)
        await cached(prompt="Explain Orchid return policy")
        kwargs = {"prompt": "Describe Orchid return policy"}
        self.observe(cached, kwargs)
        self.assertEqual(await cached(**kwargs), {"generation": 1})
        self.assertEqual(jac.last_trace.jev_calls, 1)
        self.assertEqual(await cached(**kwargs), {"generation": 1})
        self.assertEqual(jac.last_trace.jev_calls, 0)
        jac.jev = ConfigDecider("2", accept=False)
        self.observe(cached, kwargs)
        self.assertEqual(await cached(**kwargs), {"generation": 2})
        self.assertEqual(jac.last_trace.jev_calls, 1)
        self.assertEqual(jac.last_trace.llm_calls, 1)
        self.assertEqual(len(jac.jev.calls), 1)

    async def test_concurrent_semantic_misses_make_one_decision_and_no_generation(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        class DelayedDecider(FakeDecider):
            async def decide(self, request, candidates, kind):
                self.calls.append((request, candidates, kind))
                entered.set()
                await release.wait()
                return Decision(candidates[0].id, {}, self.model, 0)

        decider = DelayedDecider()
        jac, cached, _, generated = await self.semantic_cache(decider)
        kwargs = {"prompt": "Describe Orchid return policy", "model": "fixture"}
        self.observe(cached, kwargs)

        async def call_and_trace():
            value = await cached(**kwargs)
            return value, jac.last_trace

        first = asyncio.create_task(call_and_trace())
        second = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            second = asyncio.create_task(call_and_trace())
            for _ in range(20):
                await asyncio.sleep(0)
        finally:
            release.set()
        pending = [first] + ([second] if second is not None else [])
        results = await asyncio.wait_for(asyncio.gather(*pending), timeout=1)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(decider.calls), 1)
        self.assertEqual(len(generated), 1)
        self.assertEqual(sum(trace.jev_calls for _, trace in results), 1)
        self.assertEqual(sum(trace.llm_calls for _, trace in results), 0)
        self.assertEqual(results[0][0], results[1][0])
        self.assertIsNot(results[0][0], results[1][0])

    async def test_changed_settings_and_history_cannot_reach_semantic_decider(self):
        decider = FakeDecider()
        jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=0
        ))
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.030)
            return {"answer": "generated"}

        cached = jac.wrap(generate)
        base = {"model": "fixture", "messages": [
            {"role": "system", "content": "Reply in English"},
            {"role": "assistant", "content": "Earlier reply"},
            {"role": "user", "content": "Explain Orchid return policy"},
        ]}
        await cached(**base)
        changes = []
        for field, replacement in (("model", "another-model"),
                                   ("temperature", 0.5)):
            changes.append(base | {field: replacement})
        for index, content in ((0, "Reply in German"), (1, "Different earlier reply")):
            changed = copy.deepcopy(base)
            changed["messages"][index]["content"] = content
            changes.append(changed)
        for kwargs in changes:
            kwargs["messages"] = copy.deepcopy(kwargs["messages"])
            kwargs["messages"][-1]["content"] = "Describe Orchid return policy"
            self.observe(cached, kwargs)
            await cached(**kwargs)
        self.assertEqual(decider.calls, [])
        self.assertEqual(len(calls), 5)

    async def test_prompt_input_messages_and_positional_text_support_semantic_reuse(self):
        forms = [
            (("Explain Orchid return policy",), {},
             ("Describe Orchid return policy",), {}),
            ((), {"input": "Explain Orchid return policy", "model": "fixture"},
             (), {"input": "Describe Orchid return policy", "model": "fixture"}),
            ((), {"messages": [{"role": "user", "content": "Explain Orchid return policy"}]},
             (), {"messages": [{"role": "user", "content": "Describe Orchid return policy"}]}),
        ]
        for old_args, old_kwargs, new_args, new_kwargs in forms:
            with self.subTest(form=old_kwargs or old_args):
                decider = FakeDecider()
                jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
                    min_samples=20, overhead_ms=0
                ))
                calls = []

                async def generate(*args, **kwargs):
                    calls.append((args, kwargs))
                    await asyncio.sleep(0.030)
                    return {"answer": "a reusable response"}

                cached = jac.wrap(generate)
                first = await cached(*old_args, **old_kwargs)
                for _ in range(20):
                    cached.observe_reuse(new_kwargs, request_args=new_args,
                                         valid_hit=True, decision_ms=0)
                second = await cached(*new_args, **new_kwargs)
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(decider.calls), 1)
                self.assertEqual(jac.last_trace.llm_calls, 0)

    async def test_semantic_scope_dependency_and_policy_guards_are_deterministic(self):
        for field, value in (("scope", "tenant-b"), ("revision", "2"), ("policy", "2")):
            with self.subTest(field=field):
                state = {"scope": "tenant-a", "revision": "1", "policy": "1"}
                decider = FakeDecider()
                jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
                    min_samples=20, overhead_ms=0
                ), context=lambda args, kwargs: Context(
                    state["scope"], {"document": state["revision"]}, state["policy"]
                ))
                calls = []

                async def generate(**kwargs):
                    calls.append(kwargs)
                    await asyncio.sleep(0.030)
                    return {"answer": "fresh response"}

                cached = jac.wrap(generate)
                await cached(prompt="Explain Orchid return policy")
                state[field] = value
                kwargs = {"prompt": "Describe Orchid return policy"}
                self.observe(cached, kwargs)
                await cached(**kwargs)
                self.assertEqual(len(calls), 2)
                self.assertEqual(decider.calls, [])
                self.assertEqual(jac.last_trace.jev_calls, 0)

    async def test_semantic_tool_action_candidate_is_never_reused(self):
        decider = FakeDecider()
        jac = self.cache(jev=decider, decision_policy=DecisionPolicy(
            min_samples=20, overhead_ms=0
        ))
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)
            await asyncio.sleep(0.030)
            return {"choices": [{"message": {"tool_calls": [
                {"id": "call-1", "function": {"name": "read", "arguments": "{}"}}
            ]}}]}

        cached = jac.wrap(generate)
        await cached(prompt="Explain Orchid return policy")
        kwargs = {"prompt": "Describe Orchid return policy"}
        self.observe(cached, kwargs)
        await cached(**kwargs)
        self.assertEqual(len(calls), 2)
        self.assertEqual(decider.calls, [])
        self.assertEqual(jac.last_trace.jev_calls, 0)


if __name__ == "__main__":
    unittest.main()
