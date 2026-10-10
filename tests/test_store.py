"""Persistence and cache-partition regression tests using only the standard library."""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jevcache.models import Context, Request, fingerprint, snapshot
from jevcache.store import Entry, SQLiteStore


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def entry(key="one", **changes):
    original = Entry(
        key=key,
        kind="answer",
        scope="tenant-a",
        task="exposure",
        value={"exposure": 42},
        request={"text": "Calculate portfolio exposure"},
        dependencies={"prices": "revision-1"},
        policy_version="1",
        output_contract="json-v1",
        created_at=1000.0,
        expires_at=1060.0,
        avoided_ms=500.0,
        metadata={"validator": "checked"},
    )
    return replace(original, **changes)


class SQLiteStoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.store = SQLiteStore(clock=self.clock)

    def tearDown(self):
        self.store.close()

    def test_round_trip_and_missing_key(self):
        artifact = entry()
        self.store.put(artifact)
        self.assertEqual(self.store.get("one"), artifact)
        self.assertIsNone(self.store.get("absent"))

    def test_expiry_is_exclusive_and_permanent_entries_remain_live(self):
        self.store.put(entry())
        self.store.put(entry("permanent", expires_at=None))
        self.clock.now = 1059.999
        self.assertIsNotNone(self.store.get("one"))
        self.clock.now = 1060.0
        self.assertIsNone(self.store.get("one"))
        self.clock.now = 1_000_000.0
        self.assertIsNotNone(self.store.get("permanent"))

    def test_json_snapshot_is_detached_from_callers(self):
        value = {"items": [1, {"position": 2}]}
        self.store.put(entry(value=value))
        value["items"][1]["position"] = 99
        first = self.store.get("one")
        self.assertEqual(first.value["items"][1]["position"], 2)
        first.value["items"].append(3)
        self.assertEqual(self.store.get("one").value["items"], [1, {"position": 2}])

    def test_upsert_replaces_artifact_and_search_text(self):
        self.store.put(entry(request={"text": "stale token"}))
        self.store.put(entry(value=7, request={"text": "current token"}))
        self.assertEqual(self.store.get("one").value, 7)
        candidates = self.store.candidates(
            kind="answer", scope="tenant-a", task="exposure", text="current"
        )
        self.assertEqual([item.key for item in candidates], ["one"])
        if self.store.has_fts:
            self.assertEqual(
                self.store.candidates(
                    kind="answer", scope="tenant-a", task="exposure", text="stale"
                ),
                [],
            )

    def test_candidates_cannot_cross_kind_scope_task_or_expiry(self):
        for artifact in (
            entry(),
            entry("other-tenant", scope="tenant-b"),
            entry("other-task", task="returns"),
            entry("other-kind", kind="plan"),
            entry("expired", expires_at=999.0),
        ):
            self.store.put(artifact)
        result = self.store.candidates(
            kind="answer", scope="tenant-a", task="exposure", text="portfolio"
        )
        self.assertEqual([item.key for item in result], ["one"])

    def test_candidates_limit_and_punctuation_are_bounded(self):
        for index in range(8):
            self.store.put(entry(str(index)))
        result = self.store.candidates(
            kind="answer",
            scope="tenant-a",
            task="exposure",
            text='portfolio" OR *; SELECT anything',
            limit=3,
        )
        self.assertLessEqual(len(result), 3)
        self.assertEqual(
            self.store.candidates(
                kind="answer", scope="tenant-a", task="exposure", text="portfolio", limit=0
            ),
            [],
        )

    def test_fallback_retrieval_preserves_partition_and_expiry(self):
        self.store.has_fts = False
        self.store.put(entry())
        self.store.put(entry("other-tenant", scope="tenant-b"))
        self.store.put(entry("expired", expires_at=999.0))
        result = self.store.candidates(kind="answer", scope="tenant-a", task="exposure", text="")
        self.assertEqual([item.key for item in result], ["one"])

    def test_dependency_invalidation_preserves_unrelated_entries(self):
        self.store.put(entry())
        self.store.put(entry("plan", kind="plan", dependencies={}))
        self.store.put(entry("other", scope="tenant-b"))
        self.assertEqual(self.store.invalidate(scope="tenant-a", dependency="prices"), 1)
        self.assertIsNone(self.store.get("one"))
        self.assertIsNotNone(self.store.get("plan"))
        self.assertIsNotNone(self.store.get("other"))
        self.assertEqual(
            self.store.candidates(
                kind="answer", scope="tenant-a", task="exposure", text="portfolio"
            ),
            [],
        )

    def test_scope_invalidation_is_complete_and_idempotent(self):
        self.store.put(entry())
        self.store.put(entry("plan", kind="plan", dependencies={}))
        self.store.put(entry("other", scope="tenant-b"))
        self.assertEqual(self.store.invalidate(scope="tenant-a"), 2)
        self.assertEqual(self.store.invalidate(scope="tenant-a"), 0)
        self.assertIsNotNone(self.store.get("other"))

    def test_observations_are_labelled_and_separate_by_cohort(self):
        self.assertEqual(self.store.outcome("absent"), (0, 0, 0.0))
        self.store.observe_reuse("exposure", valid_hit=True, decision_ms=100)
        self.store.observe_reuse("exposure", valid_hit=False, decision_ms=300)
        self.store.observe_reuse("other", valid_hit=True, decision_ms=900)
        self.assertEqual(self.store.outcome("exposure"), (2, 1, 200.0))
        self.assertEqual(self.store.outcome("other"), (1, 1, 900.0))
        for label in (1, 0, None, "true", 0.999):
            with self.subTest(label=label), self.assertRaises(TypeError):
                self.store.observe_reuse("exposure", valid_hit=label, decision_ms=10)

    def test_invalid_latency_and_expiry_numbers_are_rejected(self):
        for value in (-1, math.nan, math.inf, -math.inf, True, False, "10", None):
            with self.subTest(avoided=value), self.assertRaises(ValueError):
                self.store.put(entry(avoided_ms=value))
            with self.subTest(observation=value), self.assertRaises(ValueError):
                self.store.observe_reuse("exposure", valid_hit=True, decision_ms=value)
        for value in (math.nan, math.inf, -math.inf, True, False, "10"):
            with self.subTest(expiry=value), self.assertRaises(ValueError):
                self.store.put(entry(expires_at=value))

    def test_traces_preserve_order_without_mutation(self):
        trace = {"path": "miss", "events": [{"status": "executed"}]}
        self.store.record_trace(trace)
        trace["events"][0]["status"] = "changed"
        self.store.record_trace({"path": "full_hit", "events": []})
        self.assertEqual(
            self.store.traces(),
            [
                {"path": "miss", "events": [{"status": "executed"}]},
                {"path": "full_hit", "events": []},
            ],
        )

    def test_artifacts_observations_and_traces_survive_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            with SQLiteStore(path, clock=self.clock) as durable:
                durable.put(entry())
                durable.observe_reuse("cohort", valid_hit=True, decision_ms=50)
                durable.record_trace({"path": "miss"})
            with SQLiteStore(path, clock=self.clock) as reopened:
                self.assertEqual(reopened.get("one"), entry())
                self.assertEqual(reopened.outcome("cohort"), (1, 1, 50.0))
                self.assertEqual(reopened.traces(), [{"path": "miss"}])


class InputContractTests(unittest.TestCase):
    def test_fingerprint_normalizes_object_order_but_preserves_types_and_array_order(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(fingerprint(True), fingerprint(1))
        self.assertNotEqual(fingerprint([1, 2]), fingerprint([2, 1]))

    def test_ambiguous_or_nonfinite_json_cannot_be_fingerprinted(self):
        for value in ({1: "number-key"}, {"nested": math.nan}, object()):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                fingerprint(value)

    def test_request_and_context_require_explicit_nonempty_identifiers(self):
        for request in (
            Request("", {}, Context("tenant")),
            Request("task", {}, Context("")),
            Request("task", {}, Context("tenant", policy_version="")),
            Request("task", {}, Context("tenant"), output_contract=""),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                request.to_dict()

    def test_request_and_context_identifiers_are_strings(self):
        for invalid in (1, True, [], {}):
            for request in (
                Request(invalid, {}, Context("tenant")),
                Request("task", {}, Context(invalid)),
                Request("task", {}, Context("tenant", policy_version=invalid)),
                Request("task", {}, Context("tenant"), output_contract=invalid),
            ):
                with self.subTest(request=request), self.assertRaises(TypeError):
                    request.to_dict()

    def test_dependencies_are_explicit_revisions_and_snapshots_are_independent(self):
        with self.assertRaises(TypeError):
            Context("tenant", {"prices": 1}).to_dict()
        original = {"items": [1]}
        copied = snapshot(original)
        copied["items"].append(2)
        self.assertEqual(original, {"items": [1]})


if __name__ == "__main__":
    unittest.main()
