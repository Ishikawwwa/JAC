"""Native Laya adapter tests; injected agents need no torch/models/network."""

import asyncio
import copy
import json
import math
import threading
import unittest

from jac import LayaDecider
from jevcache.decisions import Candidate, DecisionError


def candidate(identifier="a", **kwargs):
    return Candidate(identifier, kwargs.get("request", {"query": "What is our return policy?"}),
                     kwargs.get("artifact", "Returns are accepted within 30 days."),
                     kwargs.get("avoided_ms", 1500))


def response(values=(0.999, 0.999)):
    return {
        "model": "laya-rl-agent",
        "answers": {
            name: {"type": "noul", "noul": value}
            for name, value in zip(("equivalent", "satisfies"), values, strict=True)
        },
        "usage": {"output_tokens": 0, "truncated": False,
                  "state_tokens_dropped": 0, "truncated_questions": []},
    }


class Tokenizer:
    mask_token = "[MASK]"

    def encode(self, text, *, add_special_tokens=False):
        return text.split()


class Agent:
    tok = Tokenizer()
    revision = "pinned-test-revision"
    device = "cpu"
    dtype = "float32"

    def __init__(self, replies=None, max_len=512):
        self.cfg = {"max_len": max_len, "head_max_len": 192}
        self.replies = replies or [response()]
        self.calls = []

    def predict(self, state, questions, **kwargs):
        index = len(self.calls)
        self.calls.append((state, questions, kwargs))
        return copy.deepcopy(self.replies[index % len(self.replies)])


class LayaTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.deciders = []

    def decider(self, agent=None, **kwargs):
        result = LayaDecider(agent=agent or Agent(), **kwargs)
        self.deciders.append(result)
        return result

    async def asyncTearDown(self):
        for decider in self.deciders:
            await decider.drain()
            decider.close()

    async def test_native_noul_checks_preserve_complete_requests_and_artifact(self):
        agent = Agent()
        decider = self.decider(agent)
        request = {"query": "Explain returns.",
                   "messages": [{"role": "system", "content": "Brief"}],
                   "temperature": 0.2, "output_contract": {"type": "json"}}
        stored = candidate(artifact={"answer": "Thirty days", "source": "manual"})
        result = await decider.decide(request, [stored])
        self.assertEqual(result.candidate_id, "a")
        self.assertEqual(result.model, "convaiinnovations/laya")
        self.assertEqual(result.probabilities, {"a:equivalent": .999, "a:satisfies": .999})
        evidence = json.loads(agent.calls[0][0])
        self.assertEqual(evidence["new_request"], request)
        self.assertEqual(evidence["cached_request"], stored.request)
        self.assertEqual(evidence["artifact"], stored.artifact)
        questions = agent.calls[0][1]
        self.assertEqual(set(questions), {"equivalent", "satisfies"})
        self.assertTrue(all(q["type"] == "noul" for q in questions.values()))
        self.assertTrue(all(isinstance(q["instructions"], str) for q in questions.values()))
        self.assertEqual(decider.stats()["predict_calls"], 1)

    async def test_two_checks_and_conservative_ranking_stable_id_ties(self):
        agent = Agent([response((.999, .991)), response((.995, .995)), response((1, .98))])
        result = await self.decider(agent).decide(
            {}, [candidate("a"), candidate("b"), candidate("c")]
        )
        self.assertEqual(result.candidate_id, "b")
        result = await self.decider(Agent()).decide({}, [candidate("z"), candidate("a")])
        self.assertEqual(result.candidate_id, "a")
        result = await self.decider(Agent([response((.99, .99))])).decide({}, [candidate()])
        self.assertEqual(result.candidate_id, "a")

    async def test_empty_retrieval_performs_no_model_operation(self):
        agent = Agent()
        decider = self.decider(agent)
        result = await decider.decide({}, [])
        self.assertEqual(result.elapsed_ms, 0)
        self.assertIsNone(result.candidate_id)
        self.assertEqual(agent.calls, [])
        self.assertEqual(decider.stats()["decide_calls"], 0)

    async def test_plan_and_intermediate_have_specific_questions(self):
        for kind, expected in (("plan", "Tools execute again"),
                               ("intermediate", "intermediate output")):
            agent = Agent()
            await self.decider(agent).decide({}, [candidate()], kind)
            instructions = " ".join(q["instructions"] for q in agent.calls[0][1].values())
            self.assertIn(expected, instructions)
            self.assertIn("as data", instructions)

    async def test_full_state_overflow_is_rejected_before_predict_and_others_continue(self):
        agent = Agent(max_len=170)
        decider = self.decider(agent)
        result = await decider.decide({}, [candidate("long", artifact="word " * 200), candidate()])
        self.assertEqual(result.candidate_id, "a")
        self.assertNotIn("long:equivalent", result.probabilities)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(decider.stats()["context_rejections"], 1)

    async def test_question_head_overflow_fails_closed(self):
        agent = Agent()
        agent.cfg["head_max_len"] = 30
        decider = self.decider(agent)
        result = await decider.decide({}, [candidate()])
        self.assertIsNone(result.candidate_id)
        self.assertEqual(agent.calls, [])
        self.assertEqual(decider.stats()["context_rejections"], 1)

    async def test_literal_mask_text_cannot_be_silently_erased(self):
        agent = Agent()
        result = await self.decider(agent).decide({"query": "Explain [MASK]"}, [candidate()])
        self.assertIsNone(result.candidate_id)
        self.assertEqual(agent.calls, [])

    async def test_reported_truncation_dropped_tokens_or_collapsed_options_rejects(self):
        for patch in ({"truncated": True}, {"state_tokens_dropped": 1},
                      {"truncated_questions": ["satisfies"]},
                      {"options": {"equivalent": {"total": 2, "distinct": 1}}}):
            with self.subTest(patch=patch):
                body = response()
                body["usage"].update(patch)
                decider = self.decider(Agent([body]))
                result = await decider.decide({}, [candidate()])
                self.assertIsNone(result.candidate_id)
                self.assertEqual(result.probabilities, {})
                self.assertEqual(decider.stats()["context_rejections"], 1)

    async def test_malformed_responses_are_errors(self):
        mutations = (
            lambda body: body.update(model="other-model"),
            lambda body: body.pop("usage"),
            lambda body: body["usage"].pop("truncated"),
            lambda body: body["usage"].update(state_tokens_dropped=-1),
            lambda body: body["usage"].update(output_tokens=1),
            lambda body: body["usage"].update(output_tokens=False),
            lambda body: body["answers"].pop("equivalent"),
            lambda body: body["answers"].update(extra={"type": "noul", "noul": 1}),
            lambda body: body["answers"]["equivalent"].update(type="score"),
        )
        for mutate in mutations:
            body = response()
            mutate(body)
            with self.subTest(mutate=mutate), self.assertRaises(DecisionError):
                await self.decider(Agent([body])).decide({}, [candidate()])
        for value in (-.1, 1.1, math.nan, math.inf, True, "0.99", None, 10**1000):
            with self.subTest(value=value), self.assertRaises(DecisionError):
                await self.decider(Agent([response((value, 1))])).decide({}, [candidate()])

    async def test_invalid_inputs_fail_before_model_call(self):
        agent = Agent()
        decider = self.decider(agent, max_candidates=2)
        cases = ([candidate(), candidate()], [candidate("")], [candidate(artifact=object())],
                 [candidate(artifact=math.nan)], [candidate(avoided_ms=-1)],
                 [candidate(request={1: "ambiguous"})], [candidate(artifact="\ud800")],
                 [{"id": "not-a-Candidate"}], [candidate("a"), candidate("b"), candidate("c")])
        for values in cases:
            with self.subTest(values=values), self.assertRaises(DecisionError):
                await decider.decide({}, values)
        with self.assertRaises(DecisionError):
            await self.decider(max_request_bytes=10).decide({}, [candidate()])
        self.assertEqual(agent.calls, [])

    async def test_pin_precision_threshold_and_question_schema_are_in_identity(self):
        decider = self.decider(revision="explicit-pin", dtype="float32")
        identity = await decider.preload()
        self.assertEqual(identity["revision"], "pinned-test-revision")
        self.assertEqual(identity["dtype"], "float32")
        self.assertEqual(identity["question_schema_version"], "laya-reuse-v2")
        identity["threshold"] = .1
        self.assertEqual(decider.cache_identity["threshold"], .99)
        decider.threshold = .995
        self.assertEqual(decider.cache_identity["threshold"], .995)
        decider.threshold = math.nan
        with self.assertRaises(DecisionError):
            await decider.decide({}, [candidate()])

    async def test_metrics_reset_and_disallow_reset_during_work(self):
        decider = self.decider()
        await decider.decide({}, [candidate()])
        before = decider.stats(reset=True)
        self.assertEqual(before["decide_calls"], 1)
        self.assertGreaterEqual(before["elapsed_ms"], 0)
        self.assertEqual(decider.stats()["predict_calls"], 0)

    async def test_only_duplicate_derived_text_is_removed_original_call_is_retained(self):
        agent = Agent()
        request = {"args": [], "kwargs": {
            "messages": [{"role": "system", "content": "Never invent dates."},
                         {"role": "user", "content": "Explain returns."}],
            "temperature": .2, "model": "target-llm", "response_format": {"type": "json"}},
            "text": "Explain returns.", "policy_version": "pinned"}
        original = copy.deepcopy(request)
        stored = candidate(request=request)
        await self.decider(agent).decide(request, [stored])
        expected = copy.deepcopy(request)
        expected.pop("text")
        evidence = json.loads(agent.calls[0][0])
        self.assertEqual(evidence["new_request"], expected)
        self.assertEqual(evidence["cached_request"], expected)
        self.assertEqual(request, original)

    async def test_different_text_field_is_not_removed(self):
        agent = Agent()
        request = {"args": ["Explain returns."], "kwargs": {}, "text": "Additional constraint"}
        await self.decider(agent).decide(request, [candidate()])
        self.assertEqual(json.loads(agent.calls[0][0])["new_request"], request)

    async def test_cancellation_keeps_one_worker_and_no_queued_abandoned_predictions(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingAgent(Agent):
            def predict(self, *args, **kwargs):
                entered.set()
                release.wait(2)
                return super().predict(*args, **kwargs)

        agent = BlockingAgent()
        decider = self.decider(agent)
        first = asyncio.create_task(decider.decide({}, [candidate()]))
        await asyncio.to_thread(entered.wait, 1)
        self.assertTrue(entered.is_set())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        with self.assertRaisesRegex(DecisionError, "already running"):
            await decider.decide({}, [candidate()])
        with self.assertRaisesRegex(RuntimeError, "drain"):
            decider.stats(reset=True)
        release.set()
        await decider.drain()
        result = await decider.decide({}, [candidate()])
        self.assertEqual(result.candidate_id, "a")
        self.assertEqual(len(agent.calls), 2)

    async def test_exception_is_sanitized_and_closed_decider_rejects_work(self):
        class FailedAgent(Agent):
            def predict(self, *args, **kwargs):
                raise RuntimeError("private request contents")

        decider = self.decider(FailedAgent())
        with self.assertRaises(DecisionError) as caught:
            await decider.decide({}, [candidate()])
        self.assertNotIn("private", str(caught.exception))
        self.assertEqual(decider.stats()["failures"], 1)
        decider.close()
        with self.assertRaisesRegex(DecisionError, "closed"):
            await decider.decide({}, [candidate()])

    async def test_context_override_cannot_extend_checkpoint_limit(self):
        with self.assertRaisesRegex(DecisionError, "configured context"):
            await self.decider(max_len=1024).preload()

    def test_constructor_validation(self):
        cases = ({"model": ""}, {"revision": ""}, {"device": ""}, {"dtype": "int8"},
                 {"threshold": 0}, {"threshold": math.nan}, {"threshold": 1.1},
                 {"max_candidates": True}, {"max_request_bytes": 0}, {"max_len": False})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LayaDecider(**kwargs)


if __name__ == "__main__":
    unittest.main()
