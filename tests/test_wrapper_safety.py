"""Regressions for configured callables, durable corruption, and independent callers."""

import asyncio
import functools
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from jac import JAC


class CallableIsolationTests(unittest.TestCase):
    def setUp(self):
        self.cache = JAC(":memory:")

    def tearDown(self):
        self.cache.close()

    def test_partials_with_different_models_cannot_share_an_answer(self):
        calls = []

        def llm(model, prompt):
            calls.append((model, prompt))
            return {"model": model, "answer": prompt}

        model_a = self.cache.wrap(functools.partial(llm, "model-a"))
        model_b = self.cache.wrap(functools.partial(llm, "model-b"))
        self.assertEqual(model_a("hello")["model"], "model-a")
        self.assertEqual(model_b("hello")["model"], "model-b")
        self.assertEqual(model_a("hello")["model"], "model-a")
        self.assertEqual(calls, [("model-a", "hello"), ("model-b", "hello")])

    def test_closures_with_different_configuration_cannot_share_an_answer(self):
        calls = []

        def configure(model):
            def llm(prompt):
                calls.append(model)
                return {"model": model, "answer": prompt}

            return llm

        model_a = self.cache.wrap(configure("model-a"))
        model_b = self.cache.wrap(configure("model-b"))
        model_a("hello")
        self.assertEqual(model_b("hello")["model"], "model-b")
        model_b("hello")
        self.assertEqual(calls, ["model-a", "model-b"])

    def test_bound_clients_with_different_configuration_cannot_share_an_answer(self):
        calls = []

        class Client:
            def __init__(self, model):
                self.model = model

            def create(self, prompt):
                calls.append(self.model)
                return {"model": self.model, "answer": prompt}

        model_a = self.cache.wrap(Client("model-a").create)
        model_b = self.cache.wrap(Client("model-b").create)
        model_a("hello")
        self.assertEqual(model_b("hello")["model"], "model-b")
        self.assertEqual(calls, ["model-a", "model-b"])

    def test_corrupt_durable_metadata_falls_back_to_generation(self):
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return {"generation": len(calls)}

        cached = self.cache.wrap(llm)
        self.assertEqual(cached("hello"), {"generation": 1})
        row = self.cache.store.connection.execute(
            "SELECT key, data FROM entries WHERE kind = 'llm_answer'"
        ).fetchone()
        data = json.loads(row["data"])
        data["metadata"] = None
        self.cache.store.connection.execute(
            "UPDATE entries SET data = ? WHERE key = ?", (json.dumps(data), row["key"])
        )
        self.cache.store.connection.commit()
        self.assertEqual(cached("hello"), {"generation": 2})
        self.assertEqual(calls, ["hello", "hello"])

    def test_sync_call_preserves_the_callers_thread_local_client(self):
        sessions = threading.local()
        sessions.client = object()
        caller_thread = threading.get_ident()

        def llm(prompt):
            return {
                "has_client": sessions.client is not None,
                "thread": threading.get_ident(),
                "answer": prompt,
            }

        cached = self.cache.wrap(llm)
        self.assertEqual(
            cached("hello"), {"has_client": True, "thread": caller_thread, "answer": "hello"}
        )
        self.assertEqual(cached("hello")["thread"], caller_thread)

    def test_write_contention_keeps_hits_and_misses_fast_and_preserves_provider_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite"
            cache = JAC(path)
            other = sqlite3.connect(path)
            calls = []
            caller_thread = threading.get_ident()

            def llm(prompt):
                calls.append((prompt, threading.get_ident()))
                return {"answer": prompt}

            try:
                cached = cache.wrap(llm)
                self.assertEqual(cached("warm"), {"answer": "warm"})
                other.execute("BEGIN IMMEDIATE")
                for prompt, generations in (("warm", 0), ("miss", 1), ("miss", 1)):
                    with self.subTest(prompt=prompt, generations=generations):
                        started = time.perf_counter()
                        self.assertEqual(cached(prompt), {"answer": prompt})
                        elapsed = time.perf_counter() - started
                        self.assertLess(elapsed, 0.250)
                        self.assertEqual(cache.last_trace.llm_calls, generations)
                self.assertEqual(
                    calls,
                    [("warm", caller_thread), ("miss", caller_thread), ("miss", caller_thread)],
                )
            finally:
                other.rollback()
                other.close()
                cache.close()

    def test_sync_callable_returns_its_awaitable_unchanged_without_awaiting(self):
        class AwaitableResponse:
            def __init__(self):
                self.awaited = False

            def __await__(self):
                self.awaited = True
                if False:
                    yield
                return {"answer": "only when awaited"}

        response = AwaitableResponse()
        calls = []

        def llm(prompt):
            calls.append(prompt)
            return response

        cached = self.cache.wrap(llm)
        self.assertIs(cached("hello"), response)
        self.assertIs(cached("hello"), response)
        self.assertFalse(response.awaited)
        self.assertEqual(calls, ["hello", "hello"])


class IndependentRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_canceling_owner_does_not_cancel_an_independent_waiter(self):
        cache = JAC(":memory:")
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def llm(prompt):
            calls.append(prompt)
            entered.set()
            await release.wait()
            return {"answer": prompt}

        cached = cache.wrap(llm)
        first = asyncio.create_task(cached("hello"))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            second = asyncio.create_task(cached("hello"))
            for _ in range(10):
                await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            release.set()
            self.assertEqual(await asyncio.wait_for(second, timeout=1), {"answer": "hello"})
            self.assertEqual(calls, ["hello", "hello"])
        finally:
            release.set()
            await asyncio.gather(
                *[task for task in (first, second) if task is not None], return_exceptions=True
            )
            cache.close()

    async def test_canceling_waiter_leaves_the_owner_request_running(self):
        cache = JAC(":memory:")
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def llm(prompt):
            calls.append(prompt)
            entered.set()
            await release.wait()
            return {"answer": prompt}

        cached = cache.wrap(llm)
        first = asyncio.create_task(cached("hello"))
        second = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            second = asyncio.create_task(cached("hello"))
            for _ in range(10):
                await asyncio.sleep(0)
            second.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await second
            release.set()
            self.assertEqual(await asyncio.wait_for(first, timeout=1), {"answer": "hello"})
            self.assertEqual(calls, ["hello"])
        finally:
            release.set()
            await asyncio.gather(
                *[task for task in (first, second) if task is not None], return_exceptions=True
            )
            cache.close()

    async def test_sync_call_inside_running_loop_keeps_the_callers_thread(self):
        cache = JAC(":memory:")
        sessions = threading.local()
        sessions.client = object()
        caller_thread = threading.get_ident()

        def llm(prompt):
            return {
                "has_client": sessions.client is not None,
                "thread": threading.get_ident(),
                "answer": prompt,
            }

        try:
            cached = cache.wrap(llm)
            self.assertEqual(
                cached("hello"), {"has_client": True, "thread": caller_thread, "answer": "hello"}
            )
        finally:
            cache.close()

    async def test_sync_callable_returns_the_callers_future_unchanged(self):
        cache = JAC(":memory:")
        future = asyncio.get_running_loop().create_future()
        calls = []

        def llm(prompt):
            self.assertIs(asyncio.get_running_loop(), future.get_loop())
            calls.append(prompt)
            return future

        try:
            cached = cache.wrap(llm)
            self.assertIs(cached("hello"), future)
            self.assertIs(cached("hello"), future)
            self.assertFalse(future.done())
            future.set_result({"answer": "future"})
            self.assertEqual(await future, {"answer": "future"})
            self.assertEqual(calls, ["hello", "hello"])
        finally:
            cache.close()


if __name__ == "__main__":
    unittest.main()
