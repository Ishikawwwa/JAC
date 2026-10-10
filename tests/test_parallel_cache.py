"""Speculative caching races reuse against one unchanged provider invocation."""

import asyncio
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch

from jevcache.call_cache import JAC
from jevcache.decisions import Decision, DecisionPolicy
from jevcache.models import Context
from jevcache.store import Entry


class AcceptingDecider:
    model = "parallel-test-decider"

    def __init__(self):
        self.calls = 0

    async def decide(self, _request, candidates, _kind):
        self.calls += 1
        return Decision(candidates[0].id, {}, self.model, 0)


def cached_entry(jac, *, payload=None):
    now = jac.store.clock()
    return Entry(
        "cached-original",
        "llm_answer",
        "default",
        "test-partition",
        {"codec": "json-v1", "payload": payload or {"answer": "cached"}},
        {"args": [], "kwargs": {"prompt": "Original request"}, "text": "Original request"},
        {},
        "1",
        "auto",
        now,
        now + 300,
        100,
    )


class ParallelAsyncCacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.caches = []
        self.tasks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        for jac in self.caches:
            await asyncio.wait_for(jac.drain(), timeout=2)
            jac.close()

    def cache(self, **kwargs):
        jac = JAC(":memory:", mode="parallel", **kwargs)
        self.caches.append(jac)
        return jac

    def start(self, cached, jac, *args, **kwargs):
        async def invoke():
            response = await cached(*args, **kwargs)
            return response, jac.last_trace

        task = asyncio.create_task(invoke())
        self.tasks.append(task)
        return task

    async def wait(self, event):
        await asyncio.wait_for(event.wait(), timeout=1)

    async def test_semantic_winner_returns_cache_and_cancels_started_generation(self):
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()
        jac = self.cache()
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            provider_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()

        async def lookup(_call, _context, _identity, _response_type, trace):
            trace.jev_calls += 1
            await provider_started.wait()
            return cached_entry(jac)

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            response, trace = await asyncio.wait_for(self.start(cached, jac, "Equivalent"), 1)
            self.assertEqual(response, {"answer": "cached"})
            self.assertEqual(trace.path, "full_hit")
            self.assertEqual(trace.llm_calls, 1)
            self.assertEqual(trace.jev_calls, 1)
            await self.wait(provider_cancelled)
            await jac.drain()
        self.assertEqual(calls, ["Equivalent"])
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_generation_winner_does_not_wait_for_semantic_hit(self):
        lookup_started = asyncio.Event()
        lookup_cancelled = asyncio.Event()
        jac = self.cache()
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            await lookup_started.wait()
            return {"answer": "generated"}

        async def lookup(_call, _context, _identity, _response_type, trace):
            trace.jev_calls += 1
            lookup_started.set()
            try:
                await asyncio.Event().wait()
                return cached_entry(jac)
            finally:
                lookup_cancelled.set()

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            response, trace = await asyncio.wait_for(self.start(cached, jac, "New request"), 1)
            self.assertEqual(response, {"answer": "generated"})
            self.assertEqual(trace.llm_calls, 1)
            await self.wait(lookup_cancelled)
            await jac.drain()
        self.assertEqual(calls, ["New request"])
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_generation_winner_returns_before_losing_lookup_cleanup_finishes(self):
        lookup_started = asyncio.Event()
        lookup_cancelled = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()
        jac = self.cache()

        async def generate(_prompt):
            await lookup_started.wait()
            return {"answer": "generated"}

        async def lookup(*_args):
            lookup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                lookup_cancelled.set()
                await release_cleanup.wait()
                cleanup_finished.set()
                return None

        cached = jac.wrap(generate)
        try:
            with patch.object(jac, "_semantic", side_effect=lookup):
                response, _trace = await asyncio.wait_for(self.start(cached, jac, "Request"), 1)
                self.assertEqual(response, {"answer": "generated"})
                await self.wait(lookup_cancelled)
                self.assertFalse(cleanup_finished.is_set())
                release_cleanup.set()
                await jac.drain()
        finally:
            release_cleanup.set()

    async def test_cache_winner_returns_before_losing_provider_cleanup_finishes(self):
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()
        jac = self.cache()

        async def generate(_prompt):
            provider_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                await release_cleanup.wait()
                cleanup_finished.set()
                return {"answer": "late speculative response"}

        async def lookup(*_args):
            await provider_started.wait()
            return cached_entry(jac)

        cached = jac.wrap(generate)
        try:
            with patch.object(jac, "_semantic", side_effect=lookup):
                response, trace = await asyncio.wait_for(self.start(cached, jac, "Request"), 1)
                self.assertEqual(response, {"answer": "cached"})
                self.assertEqual(trace.llm_calls, 1)
                await self.wait(provider_cancelled)
                self.assertFalse(cleanup_finished.is_set())
                release_cleanup.set()
                await jac.drain()
            # The losing generation must never overwrite the winning cached response.
            response, _trace = await asyncio.wait_for(self.start(cached, jac, "Request"), 1)
            self.assertEqual(response, {"answer": "cached"})
        finally:
            release_cleanup.set()

    async def test_lookup_rejection_or_failure_uses_already_started_generation(self):
        for failure in (None, RuntimeError("lookup unavailable"), asyncio.TimeoutError()):
            with self.subTest(failure=type(failure).__name__):
                provider_started = asyncio.Event()
                lookup_finished = asyncio.Event()
                release_provider = asyncio.Event()
                jac = self.cache()
                calls = []

                async def generate(prompt):
                    calls.append(prompt)
                    provider_started.set()
                    await release_provider.wait()
                    return {"answer": "generated"}

                async def lookup(*_args):
                    await provider_started.wait()
                    lookup_finished.set()
                    if failure is not None:
                        raise failure
                    return None

                cached = jac.wrap(generate)
                with patch.object(jac, "_semantic", side_effect=lookup):
                    request = self.start(cached, jac, "Request")
                    await self.wait(lookup_finished)
                    self.assertFalse(request.done())
                    self.assertEqual(calls, ["Request"])
                    release_provider.set()
                    response, trace = await asyncio.wait_for(request, 1)
                    self.assertEqual(response, {"answer": "generated"})
                    self.assertEqual(trace.llm_calls, 1)
                    await jac.drain()
                self.assertEqual(calls, ["Request"])

    async def test_exact_io_runs_in_parallel_and_exact_hit_needs_no_jev(self):
        jac = self.cache()
        seeded = False
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()
        lookup_started = threading.Event()
        release_lookup = threading.Event()
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            if not seeded:
                return {"answer": "original"}
            provider_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                provider_cancelled.set()

        cached = jac.wrap(generate)
        self.assertEqual(await cached("Same request"), {"answer": "original"})
        await jac.drain()
        seeded = True
        original_get = jac.store.get

        def delayed_get(key):
            lookup_started.set()
            if not release_lookup.wait(timeout=2):
                raise RuntimeError("Test did not release cache lookup")
            return original_get(key)

        try:
            with (
                patch.object(jac.store, "get", side_effect=delayed_get),
                patch.object(
                    jac, "_semantic", side_effect=AssertionError("Exact hit needs no Jev")
                ),
            ):
                request = self.start(cached, jac, "Same request")
                await self.wait(provider_started)
                self.assertFalse(request.done())
                self.assertTrue(await asyncio.to_thread(lookup_started.wait, 1))
                release_lookup.set()
                response, trace = await asyncio.wait_for(request, 1)
                self.assertEqual(response, {"answer": "original"})
                self.assertEqual(trace.path, "full_hit")
                self.assertEqual(trace.jev_calls, 0)
                self.assertEqual(trace.llm_calls, 1)
                await self.wait(provider_cancelled)
                await jac.drain()
        finally:
            release_lookup.set()
        self.assertEqual(calls, ["Same request", "Same request"])

    async def test_fast_generation_returns_while_exact_lookup_io_is_still_blocked(self):
        lookup_started = threading.Event()
        release_lookup = threading.Event()
        jac = self.cache()

        def delayed_get(_key):
            lookup_started.set()
            if not release_lookup.wait(timeout=2):
                raise RuntimeError("Test did not release cache lookup")
            return None

        async def generate(_prompt):
            while not lookup_started.is_set():
                await asyncio.sleep(0)
            return {"answer": "generated without waiting for lookup"}

        cached = jac.wrap(generate)
        try:
            with patch.object(jac.store, "get", side_effect=delayed_get):
                response, trace = await asyncio.wait_for(self.start(cached, jac, "Request"), 1)
                self.assertEqual(response, {"answer": "generated without waiting for lookup"})
                self.assertEqual(trace.llm_calls, 1)
                self.assertFalse(release_lookup.is_set())
                with self.assertRaisesRegex(RuntimeError, "outstanding|background"):
                    jac.close()
                release_lookup.set()
                await jac.drain()
        finally:
            release_lookup.set()
        self.assertFalse(jac._parallel_inflight)

    async def test_background_admission_validates_detached_response_after_return(self):
        validation_started = asyncio.Event()
        release_validation = asyncio.Event()
        validation_finished = asyncio.Event()
        validated_responses = []

        async def validate(_request, response):
            validation_started.set()
            await release_validation.wait()
            validated_responses.append(response)
            validation_finished.set()
            return True

        jac = self.cache(validator=validate)

        async def generate(_prompt):
            return {"answer": ["original"]}

        cached = jac.wrap(generate)
        try:
            response = await asyncio.wait_for(cached("Request"), 1)
            self.assertEqual(response, {"answer": ["original"]})
            await self.wait(validation_started)
            self.assertFalse(validation_finished.is_set())
            response["answer"].append("caller mutation before cache write")
            with self.assertRaisesRegex(RuntimeError, "outstanding|background"):
                jac.close()
            release_validation.set()
            await jac.drain()
            self.assertEqual(validated_responses, [{"answer": ["original"]}])
            row = jac.store.connection.execute(
                "SELECT data FROM entries WHERE kind = 'llm_answer'"
            ).fetchone()
            self.assertEqual(jac.store._entry(row).value["payload"], {"answer": ["original"]})
        finally:
            release_validation.set()

    async def test_async_context_exit_drains_background_validation_and_closes(self):
        validation_started = asyncio.Event()
        release_validation = asyncio.Event()
        context_exited = asyncio.Event()
        jac = self.cache()

        async def validate(_request, _response):
            validation_started.set()
            await release_validation.wait()
            return True

        jac.validator = validate

        async def generate(_prompt):
            return {"answer": "generated"}

        cached = jac.wrap(generate)

        async def use_context():
            async with jac:
                self.assertEqual(await cached("Request"), {"answer": "generated"})
            context_exited.set()

        operation = asyncio.create_task(use_context())
        self.tasks.append(operation)
        try:
            await self.wait(validation_started)
            self.assertFalse(context_exited.is_set())
            release_validation.set()
            await asyncio.wait_for(operation, 1)
            self.assertTrue(context_exited.is_set())
            self.assertTrue(jac._closed)
            self.assertFalse(jac._pending)
        finally:
            release_validation.set()

    async def test_identical_requests_share_one_lookup_and_generation(self):
        provider_started = asyncio.Event()
        lookup_started = asyncio.Event()
        release_lookup = asyncio.Event()
        jac = self.cache()
        calls = []
        lookups = []

        async def generate(prompt):
            calls.append(prompt)
            provider_started.set()
            await asyncio.Event().wait()

        async def lookup(*_args):
            lookups.append(1)
            lookup_started.set()
            await release_lookup.wait()
            return cached_entry(jac, payload={"answer": ["cached"]})

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            first = self.start(cached, jac, "Request")
            await self.wait(provider_started)
            await self.wait(lookup_started)
            second = self.start(cached, jac, "Request")
            for _ in range(5):
                await asyncio.sleep(0)
            release_lookup.set()
            results = await asyncio.wait_for(asyncio.gather(first, second), 1)
            await jac.drain()
        self.assertEqual(calls, ["Request"])
        self.assertEqual(lookups, [1])
        self.assertEqual([result[0] for result in results], [{"answer": ["cached"]}] * 2)
        results[0][0]["answer"].append("caller mutation")
        self.assertEqual(results[1][0], {"answer": ["cached"]})
        self.assertEqual(sum(result[1].llm_calls for result in results), 1)
        self.assertEqual(results[1][1].coalesced_nodes, 1)
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_cancelling_waiter_does_not_cancel_owner_generation(self):
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()
        release_provider = asyncio.Event()
        jac = self.cache()
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            provider_started.set()
            try:
                await release_provider.wait()
                return {"answer": "generated"}
            except asyncio.CancelledError:
                provider_cancelled.set()
                raise

        async def lookup(*_args):
            return None

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            owner = self.start(cached, jac, "Request")
            await self.wait(provider_started)
            waiter = self.start(cached, jac, "Request")
            for _ in range(5):
                await asyncio.sleep(0)
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertFalse(provider_cancelled.is_set())
            release_provider.set()
            response, trace = await asyncio.wait_for(owner, 1)
            self.assertEqual(response, {"answer": "generated"})
            self.assertEqual(trace.llm_calls, 1)
            await jac.drain()
        self.assertEqual(calls, ["Request"])
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_cancelling_owner_cleans_race_and_waiter_can_retry(self):
        first_provider_started = asyncio.Event()
        first_lookup_started = asyncio.Event()
        second_provider_started = asyncio.Event()
        first_provider_cancelled = asyncio.Event()
        first_lookup_cancelled = asyncio.Event()
        release_second_provider = asyncio.Event()
        jac = self.cache()
        calls = []
        lookup_calls = []

        async def generate(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                first_provider_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    first_provider_cancelled.set()
            second_provider_started.set()
            await release_second_provider.wait()
            return {"answer": "retry"}

        async def lookup(*_args):
            lookup_calls.append(1)
            if len(lookup_calls) == 1:
                first_lookup_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    first_lookup_cancelled.set()
            return None

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            owner = self.start(cached, jac, "Request")
            await self.wait(first_provider_started)
            await self.wait(first_lookup_started)
            waiter = self.start(cached, jac, "Request")
            for _ in range(5):
                await asyncio.sleep(0)
            owner.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await owner
            await self.wait(first_provider_cancelled)
            await self.wait(first_lookup_cancelled)
            await self.wait(second_provider_started)
            release_second_provider.set()
            response, trace = await asyncio.wait_for(waiter, 1)
            self.assertEqual(response, {"answer": "retry"})
            self.assertEqual(trace.llm_calls, 1)
            await jac.drain()
        self.assertEqual(calls, ["Request", "Request"])
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_provider_error_can_be_recovered_by_pending_cache_hit(self):
        provider_failed = asyncio.Event()
        release_lookup = asyncio.Event()
        jac = self.cache()

        async def generate(_prompt):
            provider_failed.set()
            raise RuntimeError("provider unavailable")

        async def lookup(*_args):
            await release_lookup.wait()
            return cached_entry(jac)

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            request = self.start(cached, jac, "Request")
            await self.wait(provider_failed)
            self.assertFalse(request.done())
            release_lookup.set()
            response, trace = await asyncio.wait_for(request, 1)
            self.assertEqual(response, {"answer": "cached"})
            self.assertEqual(trace.path, "full_hit")
            self.assertEqual(trace.llm_calls, 1)
            await jac.drain()

    async def test_provider_error_and_cache_miss_preserve_original_exception(self):
        lookup_started = asyncio.Event()
        provider_error = RuntimeError("provider unavailable")
        jac = self.cache()
        calls = []

        async def generate(prompt):
            calls.append(prompt)
            await lookup_started.wait()
            raise provider_error

        async def lookup(*_args):
            lookup_started.set()
            return None

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=lookup):
            with self.assertRaises(RuntimeError) as caught:
                await cached("Request")
            self.assertIs(caught.exception, provider_error)
            await jac.drain()
        self.assertEqual(calls, ["Request"])
        self.assertEqual(jac.last_trace.path, "error")
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_scope_dependency_policy_and_settings_guards_still_apply(self):
        current = {"scope": "tenant-a", "revision": "1", "policy": "1"}
        decider = AcceptingDecider()
        jac = self.cache(
            jev=decider,
            decision_policy=DecisionPolicy(min_samples=1, overhead_ms=0),
            context=lambda _args, _kwargs: Context(
                current["scope"], {"source": current["revision"]}, current["policy"]
            ),
        )
        calls = []
        generation_gate = None

        async def generate(**kwargs):
            calls.append(kwargs)
            if generation_gate is not None:
                await generation_gate.wait()
            return {"generation": len(calls)}

        cached = jac.wrap(generate)
        baseline = {"prompt": "Explain policy within 30 days", "model": "fixture"}
        original = await cached(**baseline)
        await jac.drain()
        row = jac.store.connection.execute(
            "SELECT data FROM entries WHERE kind = 'llm_answer'"
        ).fetchone()
        jac.store.put(replace(jac.store._entry(row), avoided_ms=1000))
        original_semantic = jac._semantic

        async def checked_lookup(*args):
            try:
                return await original_semantic(*args)
            finally:
                generation_gate.set()

        variants = [
            ({"scope": "tenant-b"}, {}),
            ({"revision": "2"}, {}),
            ({"policy": "2"}, {}),
            ({}, {"model": "other-model"}),
            ({}, {"prompt": "Explain policy within 90 days"}),
        ]
        for context_change, request_change in variants:
            with self.subTest(context=context_change, request=request_change):
                current.update(scope="tenant-a", revision="1", policy="1")
                current.update(context_change)
                request = baseline | request_change
                cached.observe_reuse(request, valid_hit=True, decision_ms=0)
                generation_gate = asyncio.Event()
                with patch.object(jac, "_semantic", side_effect=checked_lookup):
                    self.assertNotEqual(await asyncio.wait_for(cached(**request), 1), original)
                await jac.drain()
        self.assertEqual(decider.calls, 0)
        self.assertEqual(len(calls), 6)

    async def test_streaming_and_unsupported_requests_bypass_parallel_lookup(self):
        jac = self.cache()
        opaque = object()
        streams = []
        calls = []

        async def generate(prompt, **kwargs):
            calls.append((prompt, kwargs))
            result = object()
            streams.append(result)
            return result

        cached = jac.wrap(generate)
        with patch.object(jac, "_semantic", side_effect=AssertionError("Must bypass")):
            for _ in range(2):
                self.assertIs(await cached("Stream", stream=True), streams[-1])
                self.assertIs(await cached("Opaque", attachment=opaque), streams[-1])
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(kwargs.get("attachment", opaque) is opaque for _, kwargs in calls))
        self.assertFalse(jac._inflight)
        self.assertFalse(jac._parallel_inflight)

    async def test_late_exception_from_cancelled_provider_is_consumed(self):
        provider_started = asyncio.Event()
        provider_cancelled = asyncio.Event()
        release_provider = asyncio.Event()
        jac = self.cache()
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        unhandled = []

        async def generate(_prompt):
            provider_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                provider_cancelled.set()
                await release_provider.wait()
                raise RuntimeError("Late failure from losing speculative provider")

        async def lookup(*_args):
            await provider_started.wait()
            return cached_entry(jac)

        cached = jac.wrap(generate)
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        try:
            with patch.object(jac, "_semantic", side_effect=lookup):
                response, trace = await asyncio.wait_for(self.start(cached, jac, "Request"), 1)
                self.assertEqual(response, {"answer": "cached"})
                self.assertEqual(trace.path, "full_hit")
                await self.wait(provider_cancelled)
                release_provider.set()
                await jac.drain()
                await asyncio.sleep(0)
            self.assertEqual(unhandled, [])
        finally:
            release_provider.set()
            loop.set_exception_handler(original_handler)


class ParallelSyncCacheTests(unittest.TestCase):
    def test_invalid_mode_is_rejected(self):
        for mode in ("", "speculative", None, True, []):
            with self.subTest(mode=mode):
                with self.assertRaises((TypeError, ValueError)):
                    JAC(":memory:", mode=mode)

    def test_sync_provider_runs_concurrently_on_worker_and_preserves_arguments(self):
        jac = JAC(":memory:", mode="parallel")
        lookup_started = threading.Event()
        provider_started = threading.Event()
        caller_thread = threading.get_ident()
        calls = []

        def generate(prompt, *, model):
            calls.append((prompt, model, threading.get_ident()))
            provider_started.set()
            self.assertTrue(lookup_started.wait(timeout=1))
            return {"answer": "generated"}

        async def lookup(*_args):
            lookup_started.set()
            while not provider_started.is_set():
                await asyncio.sleep(0)
            await asyncio.Event().wait()

        try:
            cached = jac.wrap(generate)
            with patch.object(jac, "_semantic", side_effect=lookup):
                self.assertEqual(cached("Request", model="fixture"), {"answer": "generated"})
                jac.drain_sync()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][:2], ("Request", "fixture"))
            self.assertNotEqual(calls[0][2], caller_thread)
            self.assertEqual(jac.last_trace.llm_calls, 1)
            self.assertFalse(jac._inflight)
            self.assertFalse(jac._parallel_inflight)
        finally:
            jac.close()

    def test_sync_cache_hit_returns_while_uncancellable_provider_thread_keeps_running(self):
        jac = JAC(":memory:", mode="parallel")
        provider_started = threading.Event()
        release_provider = threading.Event()
        provider_finished = threading.Event()
        call_returned = threading.Event()
        responses = []
        errors = []

        def generate(_prompt):
            provider_started.set()
            try:
                if not release_provider.wait(timeout=3):
                    raise RuntimeError("Test did not release provider")
                return {"answer": "late speculative response"}
            finally:
                provider_finished.set()

        async def lookup(*_args):
            while not provider_started.is_set():
                await asyncio.sleep(0)
            return cached_entry(jac)

        cached = jac.wrap(generate)

        def invoke():
            try:
                responses.append(cached("Request"))
            except BaseException as exc:
                errors.append(exc)
            finally:
                call_returned.set()

        caller = threading.Thread(target=invoke)
        try:
            with patch.object(jac, "_semantic", side_effect=lookup):
                caller.start()
                self.assertTrue(provider_started.wait(timeout=1))
                self.assertTrue(call_returned.wait(timeout=1))
                self.assertFalse(provider_finished.is_set())
                self.assertEqual(errors, [])
                self.assertEqual(responses, [{"answer": "cached"}])
                release_provider.set()
                self.assertTrue(provider_finished.wait(timeout=1))
                caller.join(timeout=1)
                jac.drain_sync()
            self.assertFalse(jac._inflight)
            self.assertFalse(jac._parallel_inflight)
        finally:
            release_provider.set()
            if caller.is_alive():
                caller.join(timeout=2)
            jac.drain_sync()
            jac.close()

    def test_sync_bypasses_stream_opaque_cycle_and_nan_on_original_caller_thread(self):
        jac = JAC(":memory:", mode="parallel")
        caller_thread = threading.get_ident()
        opaque = object()
        cyclic = {}
        cyclic["self"] = cyclic
        cases = [
            ("Stream", {"stream": True}),
            ("Opaque", {"attachment": opaque}),
            ("Cycle", {"attachment": cyclic}),
            ("NaN", {"temperature": float("nan")}),
        ]
        calls = []
        response = object()

        def generate(prompt, **kwargs):
            calls.append((prompt, kwargs, threading.get_ident()))
            return response

        cached = jac.wrap(generate)
        try:
            with patch.object(jac, "_semantic", side_effect=AssertionError("Must bypass")):
                for prompt, kwargs in cases:
                    with self.subTest(prompt=prompt):
                        self.assertIs(cached(prompt, **kwargs), response)
                        self.assertEqual(calls[-1][2], caller_thread)
                        if "attachment" in kwargs:
                            self.assertIs(calls[-1][1]["attachment"], kwargs["attachment"])
            self.assertEqual(len(calls), len(cases))
            self.assertFalse(jac._parallel_inflight)
        finally:
            jac.drain_sync()
            jac.close()


if __name__ == "__main__":
    unittest.main()
