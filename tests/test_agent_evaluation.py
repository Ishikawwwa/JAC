"""Realistic workflow wiring, invalidation and independently scored reuse."""

import asyncio
import unittest
from unittest.mock import patch

from jac import CallableDecider, CallableEmbedder
from jevcache.agent_evaluate import (
    FixtureDecision,
    FixtureEmbeddings,
    FixtureGeneration,
    _matches,
    agent_workload,
    response_json,
    run_agent_evaluation,
)
from jevcache.decisions import Decision, DecisionPolicy


class WorkloadTests(unittest.TestCase):
    def test_calibration_is_disjoint_and_labels_are_consistent(self):
        workload = agent_workload()
        held_out = {case.request["messages"][-1]["content"] for case in workload.sequential}
        held_out.add(workload.burst.request["messages"][-1]["content"])
        for pair in workload.calibration:
            self.assertNotIn(pair.request.request["messages"][-1]["content"], held_out)
            self.assertEqual(pair.reuse, _matches(pair.seed.expected, pair.request.expected))
        self.assertEqual(len(workload.calibration), 12)
        self.assertEqual({case.family for case in workload.seeds}, {
            "support", "invoice_extraction", "portfolio_analysis"})

    def test_response_formats_and_oracle_types_are_strict(self):
        self.assertEqual(response_json('{"total_value":4000}'), {"total_value": 4000})
        self.assertEqual(response_json({"choices": [{"message": {
            "content": '{"eligible":false}'}}]}), {"eligible": False})
        self.assertTrue(_matches({"total": 4000}, {"total": 4000.0}))
        self.assertFalse(_matches({"eligible": False}, {"eligible": 0}))
        self.assertFalse(_matches({"total": 1250.5}, {"total": "1250.50"}))
        self.assertFalse(_matches({"eligible": False}, {"eligible": False, "reason": "extra"}))
        with self.assertRaises(ValueError):
            response_json('{"eligible":false,"eligible":true}')


class AgentEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_full_matrix_reuses_work_and_preserves_invalidation(self):
        report = await run_agent_evaluation(offline=True, repeats=1)
        self.assertTrue(report["fixture"])
        runs = report["runs"]
        self.assertEqual(set(runs), {"no_cache", "exact_only", "bm25_cache_first", "bm25_parallel",
                                     "embeddings_cache_first", "embeddings_parallel"})
        baseline, exact = runs["no_cache"], runs["exact_only"]
        self.assertEqual(baseline["generation_calls"], baseline["requests"])
        self.assertLess(exact["generation_calls"], baseline["generation_calls"])
        self.assertEqual(exact["decision_calls"], 0)
        for name, run in runs.items():
            self.assertEqual(run["requests"], 22)
            self.assertEqual(run["incorrect_responses"], 0, name)
            self.assertEqual(run["unsafe_reuses"], 0, name)
            self.assertEqual(run["invalidation_failures"], 0, name)
            self.assertEqual(run["tool_calls"], baseline["tool_calls"])
            self.assertEqual(run["setup_incorrect_responses"], 0)
            self.assertEqual(run["calibration_incorrect_decisions"], 0)
            self.assertGreater(run["p50_ms"], 0)
            self.assertGreaterEqual(run["p95_ms"], run["p50_ms"])
            for row in run["rows"]:
                if row["invalidation_miss_expected"]:
                    self.assertEqual(row["trace"]["path"], "miss", (name, row["case_id"]))
            if name.endswith("cache_first"):
                # The measured savings gate may skip semantic decisions on a
                # busy CI worker. Correctness and call accounting must hold.
                self.assertLessEqual(run["generation_calls"], exact["generation_calls"])
                self.assertEqual(run["generation_cancellation_requests"], 0)
            if name.endswith("parallel"):
                self.assertGreaterEqual(run["generation_calls"], runs[name.replace(
                    "parallel", "cache_first")]["generation_calls"])
        for name in ("exact_only", "bm25_cache_first", "embeddings_cache_first"):
            burst = runs[name]["categories"]["concurrency"]
            self.assertEqual(burst["coalesced_requests"], 3)

    async def test_application_async_callbacks_keep_their_event_loop(self):
        loop = asyncio.get_running_loop()
        generator = FixtureGeneration()
        embeddings = FixtureEmbeddings()
        oracle = FixtureDecision(generator)
        encoded_batches = []

        async def generate(**kwargs):
            self.assertIs(asyncio.get_running_loop(), loop)
            return await generator.complete(**kwargs)

        async def encode(texts):
            self.assertIs(asyncio.get_running_loop(), loop)
            encoded_batches.append(texts)
            return embeddings.embed(texts)

        async def decide(request, candidates, kind):
            self.assertIs(asyncio.get_running_loop(), loop)
            decision = await oracle.decide(request, candidates, kind)
            return {candidate.id: {
                "equivalent": decision.candidate_id == candidate.id,
                "satisfies": decision.candidate_id == candidate.id,
            } for candidate in candidates}

        judge = CallableDecider(decide, identity={"model": "test-callback", "version": "1"})
        embedder = CallableEmbedder(encode, identity={"model": "test-encoder", "version": "1"})
        try:
            report = await run_agent_evaluation(generation=generate,
                generation_identity={"model": "test-generator", "version": "1"},
                decider=judge, embedder=embedder, retrieval="embeddings", mode="both")
        finally:
            await judge.drain()
            judge.close()
        self.assertFalse(report["fixture"])
        self.assertGreater(len(encoded_batches), 1)
        for run in report["runs"].values():
            self.assertEqual(run["incorrect_responses"], 0)
            self.assertEqual(run["calibration_incorrect_decisions"], 0)
            self.assertEqual(run["decision_failures"], 0)
        self.assertGreater(report["runs"]["embeddings_cache_first"][
            "embedding_http_or_batch_calls"], 0)

    async def test_native_async_embedding_protocol_is_supported(self):
        loop = asyncio.get_running_loop()
        generator, vectors = FixtureGeneration(), FixtureEmbeddings()
        calls = []

        class AsyncOnlyEmbeddings:
            cache_identity = {"model": "async-only-encoder", "version": "1"}

            async def aembed(self, texts):
                if asyncio.get_running_loop() is not loop:
                    raise RuntimeError("Wrong event loop")
                calls.append(texts)
                return vectors.embed(texts)

        report = await run_agent_evaluation(generation=generator.complete,
            generation_identity=generator.cache_identity, decider=FixtureDecision(generator),
            embedder=AsyncOnlyEmbeddings(), retrieval="embeddings", mode="cache_first")
        self.assertGreater(len(calls), 1)
        self.assertEqual(report["runs"]["embeddings_cache_first"]["incorrect_responses"], 0)

    async def test_bad_model_answers_are_not_repaired_by_the_evaluation_oracle(self):
        async def bad_generation(**_kwargs):
            return {"wrong": "answer"}

        class AcceptEverything:
            cache_identity = {"model": "unsafe-test-decider", "version": "1"}

            async def decide(self, request, candidates, kind):
                return Decision(candidates[0].id, {}, "unsafe-test-decider", 0)

        report = await run_agent_evaluation(generation=bad_generation,
            generation_identity={"model": "bad-test-generator", "version": "1"},
            decider=AcceptEverything(), retrieval="bm25", mode="cache_first")
        for run in report["runs"].values():
            self.assertEqual(run["incorrect_responses"], run["requests"])
            self.assertEqual(run["setup_incorrect_responses"], 3)
        self.assertEqual(report["runs"]["no_cache"]["unsafe_reuses"], 0)
        self.assertGreater(report["runs"]["exact_only"]["unsafe_reuses"], 0)
        semantic = report["runs"]["bm25_cache_first"]
        self.assertEqual(semantic["calibration_incorrect_decisions"], 12)
        self.assertEqual(semantic["decision_calls"], 0)

    async def test_credentials_in_provider_identity_are_rejected_before_calls(self):
        calls = []

        async def generate(**kwargs):
            calls.append(kwargs)

        generator = FixtureGeneration()
        with self.assertRaises(ValueError):
            await run_agent_evaluation(generation=generate,
                generation_identity={"model": "custom", "nested": {"api_key": "secret"}},
                decider=FixtureDecision(generator), retrieval="bm25", mode="cache_first")
        self.assertEqual(calls, [])

    async def test_wrong_reuse_is_observable_with_a_correct_generator(self):
        generator = FixtureGeneration()

        class AcceptEverything:
            cache_identity = {"model": "unsafe-test-decider", "version": "1"}

            async def decide(self, request, candidates, kind):
                return Decision(candidates[0].id, {}, "unsafe-test-decider", 0)

        # Force the bounded decision to exercise bad reuse independently of
        # machine load. The savings gate itself has separate runtime tests.
        with patch.object(DecisionPolicy, "should_decide", return_value=True):
            report = await run_agent_evaluation(generation=generator.complete,
                generation_identity=generator.cache_identity, decider=AcceptEverything(),
                retrieval="bm25", mode="cache_first")
        self.assertEqual(report["runs"]["no_cache"]["incorrect_responses"], 0)
        self.assertEqual(report["runs"]["exact_only"]["incorrect_responses"], 0)
        semantic = report["runs"]["bm25_cache_first"]
        self.assertEqual(semantic["calibration_incorrect_decisions"], 6)
        self.assertGreater(semantic["unsafe_reuses"], 0)
        self.assertEqual(semantic["invalidation_failures"], 0)

    async def test_malformed_decisions_cannot_pass_calibration(self):
        generator = FixtureGeneration()

        async def generate(**kwargs):
            return {"choices": [{"message": {"content": "{}"}}]}

        class InvalidConfidence:
            cache_identity = {"model": "invalid-confidence", "version": "1"}

            async def decide(self, request, candidates, kind):
                return Decision(candidates[0].id, {"equivalent": True}, "invalid-confidence", 0)

        report = await run_agent_evaluation(generation=generate,
            generation_identity=generator.cache_identity, decider=InvalidConfidence(),
            retrieval="bm25", mode="cache_first")
        run = report["runs"]["bm25_cache_first"]
        self.assertEqual(run["calibration_incorrect_decisions"], 12)
        self.assertTrue(all(row["error"] == "ValueError" for row in run["calibration"]))
        self.assertEqual(run["decision_calls"], 0)

    async def test_malformed_generation_retains_raw_output_for_debugging(self):
        raw = {"choices": [{"message": {"content": '{"eligible":'}}]}
        seen = []

        async def truncated_generation(**kwargs):
            return raw

        class Rejects:
            cache_identity = {"model": "test-decider", "version": "1"}

            async def decide(self, request, candidates, kind):
                seen.append(candidates[0].artifact)
                return Decision(None, {}, "test-decider", 0)

        report = await run_agent_evaluation(generation=truncated_generation,
            generation_identity={"model": "truncated-generator", "version": "1"},
            decider=Rejects(), retrieval="bm25", mode="cache_first")
        for run in report["runs"].values():
            for row in (*run["rows"], *run["seed_rows"]):
                self.assertEqual(row["observed_response"], raw)
                self.assertIsNone(row["observed_answer"])
                self.assertEqual(row["error"], "JSONDecodeError")
        self.assertEqual(len(seen), 12)
        self.assertTrue(all(artifact == raw for artifact in seen))

    async def test_calibration_timeouts_fail_closed_without_hanging(self):
        generator = FixtureGeneration()

        class NeverFinishes:
            cache_identity = {"model": "stalled-test-decider", "version": "1"}

            async def decide(self, *_args):
                await asyncio.Event().wait()

        report = await asyncio.wait_for(run_agent_evaluation(generation=generator.complete,
            generation_identity=generator.cache_identity, decider=NeverFinishes(),
            retrieval="bm25", mode="cache_first", decision_timeout_s=.002), timeout=8)
        run = report["runs"]["bm25_cache_first"]
        self.assertEqual(run["incorrect_responses"], 0)
        self.assertEqual(run["calibration_incorrect_decisions"], 12)
        self.assertTrue(all(row["error"] == "TimeoutError" for row in run["calibration"]))
        self.assertEqual(run["decision_calls"], 0)

    async def test_live_providers_required_and_invalid_options_rejected(self):
        for options in ({}, {"offline": True, "generation": lambda: None},
                        {"offline": True, "retrieval": "unknown"},
                        {"offline": True, "mode": "unknown"},
                        {"offline": True, "repeats": 0}):
            with self.assertRaises(ValueError):
                await run_agent_evaluation(**options)


if __name__ == "__main__":
    unittest.main()
