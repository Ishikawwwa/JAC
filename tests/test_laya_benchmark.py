"""Benchmark integrity checks; optional real model inference is a separate run."""

import asyncio
import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from jevcache.call_cache import JAC, _semantic_request
from jevcache.decisions import Decision
from jevcache.laya_benchmark import (
    FUNCTION_NAME,
    NAMESPACE,
    DelayedGenerator,
    fixture_response,
    make_entry,
    measure_mode,
    request_kwargs,
    run,
    score_split,
    select_threshold,
    summarize,
)
from jevcache.laya_prompts import laya_questions
from jevcache.laya_workload import ANSWERS, workload
from jevcache.models import fingerprint
from jevcache.realistic_workload import FAMILIES


class FakeEncoder:
    cache_identity = {"provider": "unit-test-only", "version": "1"}

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 32
            for token in re.findall(r"\w+", text.lower()):
                vector[sum(map(ord, token)) % 32] += 1
            norm = math.hypot(*vector)
            vectors.append([value / norm for value in vector])
        return vectors


class RejectingDecider:
    model = "unit-test-only"
    cache_identity = {"provider": "unit-test-only", "threshold": 1}

    async def decide(self, request, candidates, kind):
        return Decision(None, {candidate.id + ":" + dimension: .1
                               for candidate in candidates
                               for dimension in ("equivalent", "satisfies")}, self.model, 0)


class LayaWorkloadTests(unittest.TestCase):
    def test_all_families_have_answers_and_splits_are_disjoint(self):
        dataset = workload()
        self.assertEqual(set(ANSWERS), {family[0] for family in FAMILIES})
        cal = {entry["family"] for entry in dataset["calibration"]["entries"]}
        held = {entry["family"] for entry in dataset["held_out"]["entries"]}
        self.assertFalse(cal & held)
        self.assertEqual(len(dataset["calibration"]["queries"]), 36)
        self.assertEqual(len(dataset["held_out"]["queries"]), 80)
        self.assertEqual(len(dataset["held_out"]["entries"]), 18)
        for split in (dataset["calibration"], dataset["held_out"]):
            entries = {entry["id"]: entry for entry in split["entries"]}
            self.assertEqual(len({query["id"] for query in split["queries"]}),
                             len(split["queries"]))
            for query in split["queries"]:
                if query["expected_id"]:
                    self.assertTrue(entries[query["expected_id"]]["answer_valid"])
                if query["label"] == "invalid_artifact":
                    self.assertIsNone(query["expected_id"])
                    self.assertFalse(entries[query["family"]]["answer_valid"])

    def test_short_positive_requests_preserve_numeric_guard(self):
        dataset = workload()
        for split in (dataset["calibration"], dataset["held_out"]):
            entries = {entry["id"]: entry for entry in split["entries"]}
            for query in split["queries"]:
                if query["label"] != "positive":
                    continue
                canonical = entries[query["expected_id"]]["text"]
                def guard(text):
                    return fingerprint(_semantic_request({
                        "args": [], "kwargs": request_kwargs(text)})[1])
                self.assertEqual(guard(canonical), guard(query["text"]))

    def test_seeded_entry_matches_real_exact_cache_key_and_codec(self):
        async def check():
            dataset = workload()["held_out"]
            entry = dataset["entries"][0]
            generator = DelayedGenerator(dataset["entries"], dataset["queries"], 1)
            cache = JAC(path=":memory:", namespace=NAMESPACE, ttl_s=None)
            cache.store.put(make_entry(entry, 500))
            wrapped = cache.wrap(generator, namespace=FUNCTION_NAME)
            try:
                response = await wrapped(**request_kwargs(entry["text"]))
                self.assertEqual(response["fixture_id"], fingerprint(entry["id"]))
                self.assertEqual(cache.last_trace.path, "full_hit")
                self.assertEqual(generator.starts, 0)
            finally:
                await cache.drain()
                cache.close()
        asyncio.run(check())

    def test_answer_validity_labels_are_not_leaked_in_fixture_artifacts(self):
        dataset = workload()
        for split in (dataset["calibration"], dataset["held_out"]):
            for entry in split["entries"]:
                response = fixture_response(entry)
                self.assertEqual(len(response["fixture_id"]), 64)
                if not entry["answer_valid"]:
                    self.assertNotIn(entry["id"], json.dumps(response))


class LayaBenchmarkMetricsTests(unittest.TestCase):
    def test_jevk5_mode_uses_calibrated_decider_and_original_arguments(self):
        class Accepting:
            threshold = 1
            calls = 0
            cache_identity = {"provider": "unit-test-only"}

            async def decide(self, request, candidates, kind):
                self.calls += 1
                return Decision(candidates[0].id, {}, "injected", 0)

        decider = Accepting()
        source = workload()["held_out"]["entries"][0]
        query = {"id": "test", "text": source["text"] + " Please.",
                 "expected_id": source["id"], "label": "positive"}
        split = {"entries": [source], "queries": [query]}
        calibration = {"bm25": {"qualified": True, "threshold": .95,
                               "observations": [{"valid_hit": True, "decision_ms": 0}] * 20}}
        with tempfile.TemporaryDirectory() as tmp:
            report = asyncio.run(measure_mode(
                "jevk5_bm25_cache_first", split, calibration, decider,
                FakeEncoder(), None, 10, 10, 1, 1, Path(tmp)))
        self.assertEqual(decider.calls, 1)
        self.assertEqual(decider.threshold, .95)
        self.assertTrue(report["semantic_enabled"])
        self.assertEqual(report["summary"]["generation_starts"], 0)
        self.assertEqual(report["summary"]["unique_true_semantic_accepts"], 1)

    def test_audit_includes_all_factorized_conditions(self):
        class FourCheckDecider:
            prompt_version = "laya-reuse-v3-factorized"

            async def decide(self, request, candidates, kind):
                probabilities = {
                    candidate.id + ":" + dimension: .1 if dimension == "bindings" else .999
                    for candidate in candidates
                    for dimension in laya_questions(version=self.prompt_version)
                }
                return Decision(None, probabilities, "injected-four-check-model", 0)

        entry = workload()["held_out"]["entries"][0]
        query = {"id": "case", "text": entry["text"], "expected_id": entry["id"],
                 "label": "positive", "family": entry["family"], "category": "coding"}
        result = asyncio.run(score_split({"entries": [entry], "queries": [query]},
                                        FourCheckDecider(), FakeEncoder(), 1, 500))
        for rows in result.values():
            self.assertEqual(rows[0]["scores"][entry["id"]], .1)

    def test_threshold_uses_zero_false_accept_calibration_and_max_recall(self):
        rows = [
            {"scores": {"a": .95}, "expected_id": "a"},
            {"scores": {"b": .8}, "expected_id": "b"},
            {"scores": {"c": .75}, "expected_id": None},
        ]
        selected = select_threshold(rows)
        self.assertTrue(selected["qualified"])
        self.assertEqual(selected["threshold"], .8)
        self.assertEqual(selected["sensitivity"][2]["false_accepts"], 0)
        self.assertFalse(select_threshold([
            {"scores": {"a": .7}, "expected_id": "a"},
            {"scores": {"b": .9}, "expected_id": None},
        ])["qualified"])

    def test_accuracy_denominator_is_unique_and_speculative_starts_count(self):
        def row(key, correct, label="positive", kind="semantic"):
            return {"id": key, "label": label, "reuse_kind": kind,
                    "correct_reuse": correct, "returned_fixture_id": key if correct else "wrong",
                    "elapsed_ms": 10, "decision_ms": [5], "retrieval_ms": [1],
                    "path": "full_hit", "decision_status": "accepted", "decision_calls": 1,
                    "generation_starts": 1, "generation_completions": 0,
                    "generation_cancellations": 1}
        summary = summarize([row("p", True), row("p", True), row("n", False, "hard_negative")])
        self.assertEqual(summary["measurements"], 3)
        self.assertEqual(summary["unique_queries"], 2)
        self.assertEqual(summary["semantic_acceptance_precision"], 2 / 3)
        self.assertEqual(summary["unique_semantic_acceptance_precision"], .5)
        self.assertEqual(summary["unique_false_reuses"], 1)
        self.assertEqual(summary["generation_starts_per_request"], 1)
        self.assertEqual(summary["generation_completions"], 0)
        inconsistent = summarize([row("p", True), row("p", False)])
        self.assertEqual(inconsistent["unique_semantic_acceptance_precision"], 0)
        self.assertEqual(inconsistent["inconsistent_query_ids"], ["p"])

    def test_generator_cancellation_is_observed(self):
        async def check():
            split = workload()["held_out"]
            generator = DelayedGenerator(split["entries"], split["queries"], 1000)
            task = asyncio.create_task(generator(**request_kwargs(split["queries"][0]["text"])))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual((generator.starts, generator.completions, generator.cancellations),
                             (1, 0, 1))
        asyncio.run(check())

    def test_offline_harness_fails_closed_when_no_calibration_cutoff_qualifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result.json"
            report = asyncio.run(run(decider=RejectingDecider(), embedder=FakeEncoder(),
                                     device="fixture", generation_ms=1, repeats=1,
                                     modes=("exact_only",), output=output))
            self.assertTrue(report["completed"])
            self.assertEqual(report["runs"]["exact_only"]["summary"]["unique_false_reuses"], 0)
            self.assertEqual(report["runs"]["exact_only"]["summary"]["generation_starts"], 68)
            self.assertFalse(report["calibration"]["bm25"]["qualified"])
            shadow = report["held_out_shadow"]["bm25"]
            self.assertEqual(len(shadow["queries"]), 80)
            self.assertIsNone(shadow["frozen_configuration_quality"])
            self.assertFalse(shadow["semantic_configuration_qualified"])
            self.assertEqual(len(shadow["exploratory_cutoff_diagnostics"]), 9)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
