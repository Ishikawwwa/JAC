"""Freshness and decision provenance must survive a speculative cache race."""

import asyncio
import threading
import unittest
from unittest.mock import patch

from jevcache.call_cache import JAC, _semantic_request
from jevcache.decisions import Decision, DecisionPolicy
from jevcache.models import Context, fingerprint
from jevcache.store import Entry


class ChangingDecider:
    model = "decision-fixture"

    def __init__(self):
        self.version = "1"

    @property
    def cache_identity(self):
        return {"provider": "fixture", "model": self.model, "version": self.version}

    async def decide(self, _request, candidates, _kind):
        return Decision(candidates[0].id, {}, self.model, 0)


class ParallelValidityTests(unittest.IsolatedAsyncioTestCase):
    def cache(self, **kwargs):
        return JAC(":memory:", mode="parallel", **kwargs)

    @staticmethod
    def seed(jac, *, prompt="Same request", namespace="validity", expires_at=None):
        identity = fingerprint({"cache": jac.namespace, "function": namespace})
        call = {"args": [], "kwargs": {"prompt": prompt}}
        context = Context("default")
        key = fingerprint(
            {
                "kind": "llm_answer",
                "function": identity,
                "call": call,
                "context": context.to_dict(),
                "contract": "auto",
            }
        )
        shape = _semantic_request(call)
        partition = fingerprint({"function": identity, "guard": fingerprint(shape[1])})
        entry = Entry(
            key,
            "llm_answer",
            context.scope,
            partition,
            {"codec": "json-v1", "payload": {"answer": "cached"}},
            {**call, "text": prompt},
            {},
            context.policy_version,
            "auto",
            jac.store.clock(),
            expires_at,
            1000,
            {"semantic_identity": None},
        )
        jac.store.put(entry)
        return entry

    async def test_coalesced_hit_rechecks_expiry_after_waiter_validation(self):
        now = [0.0]
        provider_started = asyncio.Event()
        release_lookup = threading.Event()
        waiter_validating = asyncio.Event()
        release_validator = asyncio.Event()
        validation_calls = 0
        provider_calls = []

        async def validator(_request, _response):
            nonlocal validation_calls
            validation_calls += 1
            if validation_calls == 2:
                waiter_validating.set()
                await release_validator.wait()
            return True

        async def provider(*, prompt):
            provider_calls.append(prompt)
            if len(provider_calls) == 1:
                provider_started.set()
                await asyncio.Event().wait()
            return {"answer": "fresh"}

        jac = self.cache(validator=validator)
        jac.store.clock = lambda: now[0]
        self.seed(jac, expires_at=1.0)
        cached = jac.wrap(provider, namespace="validity")
        original_get = jac.store.get

        def blocked_get(key):
            if not release_lookup.wait(timeout=2):
                raise RuntimeError("The test did not release cache lookup")
            return original_get(key)

        try:
            with patch.object(jac.store, "get", side_effect=blocked_get):
                owner = asyncio.create_task(cached(prompt="Same request"))
                await asyncio.wait_for(provider_started.wait(), 1)
                waiter = asyncio.create_task(cached(prompt="Same request"))
                for _ in range(5):
                    await asyncio.sleep(0)
                release_lookup.set()
                await asyncio.wait_for(waiter_validating.wait(), 1)
                self.assertEqual(await asyncio.wait_for(owner, 1), {"answer": "cached"})
                now[0] = 2.0
                release_validator.set()
                self.assertEqual(await asyncio.wait_for(waiter, 1), {"answer": "fresh"})
                self.assertEqual(len(provider_calls), 2)
            await jac.drain()
        finally:
            release_lookup.set()
            release_validator.set()
            await jac.drain()
            jac.close()

    async def test_deferred_promotion_retains_identity_that_validated_hit(self):
        decider = ChangingDecider()
        jac = self.cache(
            jev=decider, decision_policy=DecisionPolicy(min_samples=1, overhead_ms=0)
        )
        self.seed(jac, prompt="Explain Orchid return policy")

        async def provider(**_kwargs):
            await asyncio.Event().wait()

        cached = jac.wrap(provider, namespace="validity")
        request = {"prompt": "Describe Orchid return policy"}
        cached.observe_reuse(request, valid_hit=True, decision_ms=0)
        validated_identity = decider.cache_identity
        try:
            self.assertEqual(await asyncio.wait_for(cached(**request), 1), {"answer": "cached"})
            decider.version = "2"
            await jac.drain()
            rows = jac.store.connection.execute(
                "SELECT data FROM entries WHERE kind = 'llm_answer'"
            ).fetchall()
            aliases = [
                jac.store._entry(row)
                for row in rows
                if jac.store._entry(row).metadata.get("semantic_identity") is not None
            ]
            self.assertEqual(len(aliases), 1)
            self.assertEqual(aliases[0].metadata["semantic_identity"], validated_identity)
        finally:
            await jac.drain()
            jac.close()

    async def test_delayed_admission_does_not_refresh_generated_response_ttl(self):
        now = [0.0]
        admission_started = asyncio.Event()
        release_admission = asyncio.Event()

        async def validator(_request, _response):
            admission_started.set()
            await release_admission.wait()
            return True

        async def provider(*, prompt):
            return {"answer": prompt}

        jac = self.cache(ttl_s=1, validator=validator)
        jac.store.clock = lambda: now[0]
        cached = jac.wrap(provider, namespace="validity")
        try:
            self.assertEqual(await cached(prompt="Same request"), {"answer": "Same request"})
            await asyncio.wait_for(admission_started.wait(), 1)
            now[0] = 100.0
            release_admission.set()
            await jac.drain()
            rows = jac.store.connection.execute(
                "SELECT data FROM entries WHERE kind = 'llm_answer'"
            ).fetchall()
            for row in rows:
                entry = jac.store._entry(row)
                self.assertLessEqual(entry.expires_at, 1.0)
                self.assertIsNone(jac.store.get(entry.key))
        finally:
            release_admission.set()
            await jac.drain()
            jac.close()

    async def test_close_rejects_active_coalesced_waiter_after_background_drain(self):
        provider_started = asyncio.Event()
        release_lookup = threading.Event()
        waiter_validating = asyncio.Event()
        release_validator = asyncio.Event()
        validation_calls = 0

        async def validator(_request, _response):
            nonlocal validation_calls
            validation_calls += 1
            if validation_calls == 2:
                waiter_validating.set()
                await release_validator.wait()
            return True

        async def provider(**_kwargs):
            provider_started.set()
            await asyncio.Event().wait()

        jac = self.cache(validator=validator)
        self.seed(jac)
        cached = jac.wrap(provider, namespace="validity")
        original_get = jac.store.get
        owner = waiter = None

        def blocked_get(key):
            if not release_lookup.wait(timeout=2):
                raise RuntimeError("The test did not release cache lookup")
            return original_get(key)

        try:
            with patch.object(jac.store, "get", side_effect=blocked_get):
                owner = asyncio.create_task(cached(prompt="Same request"))
                await asyncio.wait_for(provider_started.wait(), 1)
                waiter = asyncio.create_task(cached(prompt="Same request"))
                for _ in range(5):
                    await asyncio.sleep(0)
                release_lookup.set()
                await asyncio.wait_for(waiter_validating.wait(), 1)
                self.assertEqual(await asyncio.wait_for(owner, 1), {"answer": "cached"})
                await jac.drain()
                with self.assertRaisesRegex(RuntimeError, "outstanding"):
                    jac.close()
                release_validator.set()
                self.assertEqual(await asyncio.wait_for(waiter, 1), {"answer": "cached"})
                await jac.drain()
        finally:
            release_lookup.set()
            release_validator.set()
            await asyncio.gather(*(t for t in (owner, waiter) if t), return_exceptions=True)
            await jac.drain()
            jac.close()


if __name__ == "__main__":
    unittest.main()
