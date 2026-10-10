"""JevK5 wire accounting, option readout and safe reuse without model downloads."""

import asyncio
import json
import math
import threading
import unittest
from unittest.mock import patch

from jac import JAC, Candidate, JevK5Decider
from jevcache.decisions import DecisionError
from jevcache.evaluate import providers


class Endpoint:
    def __init__(self, probability=.97, **overrides):
        self.probability = probability
        self.overrides = overrides
        self.posts = []

    def post(self, path, payload):
        self.posts.append((path, payload))
        if path == "/tokenize":
            return {"tokens": [1, 2, 3]}
        output = {"truncated": False, "tokens_evaluated": 3, "tokens_predicted": 1,
                  "completion_probabilities": [{"top_logprobs": [
                      {"token": "A", "logprob": math.log(self.probability)},
                      {"token": "B", "logprob": math.log(1 - self.probability)},
                  ]}]}
        return {**output, **self.overrides}


class JevK5Tests(unittest.IsolatedAsyncioTestCase):
    def model(self, endpoint=None, **kwargs):
        model = JevK5Decider(temperature=1, threshold=.9, **kwargs)
        endpoint = endpoint or Endpoint()
        model.client.post = endpoint.post
        model._prompt = lambda state, instruction, options: json.dumps(state) + instruction
        model._options = lambda question: list(question["criteria"].items())
        self.addCleanup(model.close)
        return model, endpoint

    async def decide(self, model):
        return await model.decide({"text": "new"}, [
            Candidate("candidate", {"text": "old"}, "answer", 500)
        ])

    async def test_two_independent_checks_and_readout_accounting(self):
        model, endpoint = self.model()
        decision = await self.decide(model)
        self.assertEqual(decision.candidate_id, "candidate")
        for value in decision.probabilities.values():
            self.assertAlmostEqual(value, .97)
        self.assertEqual(model.stats()["predict_calls"], 2)
        self.assertEqual(model.stats()["readout_tokens"], 2)
        self.assertEqual(model.stats()["input_tokens"], 6)
        completions = [payload for path, payload in endpoint.posts if path == "/completion"]
        self.assertTrue(all(payload["n_predict"] == 1 and not payload["cache_prompt"]
                            for payload in completions))
        self.assertTrue(all(payload["post_sampling_probs"] is False for payload in completions))

    async def test_temperature_and_negative_question_polarity(self):
        model, _ = self.model(Endpoint(.03), prompt_version="laya-reuse-v3-conflicts")
        self.assertEqual((await self.decide(model)).candidate_id, "candidate")
        model.temperature = 2
        decision = await self.decide(model)
        self.assertIsNone(decision.candidate_id)
        self.assertLess(min(decision.probabilities.values()), .9)

    async def test_one_failed_check_rejects_candidate(self):
        endpoint = Endpoint()
        original_post = endpoint.post

        def post(path, payload):
            if path == "/completion" and len(endpoint.posts) > 2:
                endpoint.probability = .1
            return original_post(path, payload)

        model, _ = self.model(endpoint)
        model.client.post = post
        self.assertIsNone((await self.decide(model)).candidate_id)

    async def test_overlong_context_never_calls_completion(self):
        model, endpoint = self.model(max_len=3)
        decision = await self.decide(model)
        self.assertEqual(decision.probabilities, {})
        self.assertEqual([path for path, _ in endpoint.posts], ["/tokenize"])
        self.assertEqual(model.stats()["context_rejections"], 1)

    async def test_server_truncation_or_absent_metadata_rejects(self):
        for value in (True, None, "false", 0):
            with self.subTest(value=value):
                model, _ = self.model(Endpoint(truncated=value))
                self.assertEqual((await self.decide(model)).probabilities, {})

    async def test_missing_letter_is_not_replaced_by_invented_probability(self):
        model, _ = self.model(Endpoint(completion_probabilities=[{
            "top_logprobs": [{"token": "A", "logprob": -.01}]
        }]))
        self.assertIsNone((await self.decide(model)).candidate_id)
        self.assertEqual(model.stats()["missing_option_rejections"], 1)

    async def test_duplicate_decodings_of_unrelated_special_tokens_are_allowed(self):
        model, _ = self.model(Endpoint(completion_probabilities=[{
            "top_logprobs": [{"token": "A", "logprob": math.log(.97)},
                             {"token": "B", "logprob": math.log(.03)},
                             {"token": "", "logprob": -10},
                             {"token": "", "logprob": -11}]
        }]))
        self.assertEqual((await self.decide(model)).candidate_id, "candidate")

    async def test_malformed_probabilities_and_token_counts_raise(self):
        overrides = [
            {"tokens_evaluated": 2}, {"tokens_predicted": True},
            {"completion_probabilities": []},
            {"completion_probabilities": [{"top_logprobs": [
                {"token": "A", "logprob": float("nan")}
            ]}]},
            {"completion_probabilities": [{"top_logprobs": [
                {"token": "A", "logprob": -.1}, {"token": "A", "logprob": -.2}
            ]}]},
        ]
        for override in overrides:
            with self.subTest(override=override):
                model, _ = self.model(Endpoint(**override))
                with self.assertRaises(DecisionError):
                    await self.decide(model)

    async def test_cancellation_stops_remaining_questions_and_prevents_queue(self):
        model, endpoint = self.model()
        entered, release = threading.Event(), threading.Event()
        original_post = endpoint.post

        def blocking_post(path, payload):
            if path == "/completion":
                entered.set()
                release.wait(2)
            return original_post(path, payload)

        model.client.post = blocking_post
        task = asyncio.create_task(self.decide(model))
        await asyncio.to_thread(entered.wait, 1)
        self.assertTrue(entered.is_set())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(DecisionError):
            await self.decide(model)
        release.set()
        await model.drain()
        self.assertEqual(model.stats()["predict_calls"], 1)

    async def test_exact_hit_needs_no_decision_dependency_or_endpoint(self):
        model = JevK5Decider()
        self.addCleanup(model.close)
        calls = []

        async def generate(text):
            calls.append(text)
            return "answer"

        async with JAC(path=":memory:", jev=model) as cache:
            wrapped = cache.wrap(generate)
            self.assertEqual(await wrapped("hello"), "answer")
            self.assertEqual(await wrapped("hello"), "answer")
        self.assertEqual(calls, ["hello"])
        self.assertEqual(model.stats()["predict_calls"], 0)

    def test_identity_separates_model_precision_temperature_and_prompt(self):
        model, _ = self.model(revision="pinned-commit")
        first = model.cache_identity
        model.temperature = 1.22
        self.assertNotEqual(first, model.cache_identity)
        self.assertEqual(first["revision"], "pinned-commit")
        self.assertEqual(first["provider"], "jevk5-gguf")

    def test_provider_config_forwards_jevk5_settings(self):
        settings = {"provider": "jevk5", "revision": "pinned", "temperature": 1.22,
                    "base_url": "http://127.0.0.1:8080", "threshold": .95,
                    "prompt_version": "laya-reuse-v3-bindings"}
        with patch("jevcache.evaluate.CompatibleClient"), \
                patch("jevcache.jevk5.JevK5Decider") as constructor:
            providers({"generation": {"model": "generator"}, "decision": settings})
        constructor.assert_called_once_with(**{k: v for k, v in settings.items()
                                              if k != "provider"})


if __name__ == "__main__":
    unittest.main()
