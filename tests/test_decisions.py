"""Offline correctness and latency-gate tests; no paid API is used."""

import asyncio
import json
import math
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from jevcache.decisions import (
    Candidate,
    DecisionError,
    DecisionPolicy,
    JevDecider,
    OutcomeStats,
)


def candidate(identifier="entry-a", **kwargs):
    return Candidate(
        id=identifier,
        request=kwargs.get("request", {"query": "What is our return policy?"}),
        artifact=kwargs.get("artifact", "Returns are accepted within 30 days."),
        avoided_ms=kwargs.get("avoided_ms", 1500),
    )


def response_for(payload, values=(0.999, 0.999)):
    return {
        "model": payload["model"],
        "answers": {
            key: {"type": "noul", "noul": values[index % len(values)]}
            for index, key in enumerate(payload["questions"])
        },
    }


class DecisionPolicyTests(unittest.TestCase):
    def test_requires_measured_samples_even_with_large_expected_savings(self):
        policy = DecisionPolicy()
        self.assertFalse(policy.should_decide(100_000, OutcomeStats()))
        self.assertFalse(policy.should_decide(100_000, OutcomeStats(19, 19)))
        self.assertTrue(policy.should_decide(1000, OutcomeStats(20, 20)))

    def test_uses_empirical_rate_and_strict_break_even(self):
        policy = DecisionPolicy(min_samples=10, overhead_ms=300)
        stats = OutcomeStats(attempts=100, accepted=25)
        self.assertFalse(policy.should_decide(1200, stats))
        self.assertTrue(policy.should_decide(1201, stats))
        self.assertFalse(policy.should_decide(1600, stats, retrieval_ms=100))
        self.assertTrue(policy.should_decide(1601, stats, retrieval_ms=100))
        self.assertFalse(policy.should_decide(100_000, OutcomeStats(100, 0)))

    def test_rejects_invalid_count_and_latency_data(self):
        for counts in ((-1, 0), (1, 2), (True, 1), (1.5, 1)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                OutcomeStats(*counts)
        for value in (-1, math.nan, math.inf, True, "30"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DecisionPolicy().should_decide(value, OutcomeStats(20, 20))
        with self.assertRaises(ValueError):
            DecisionPolicy(min_samples=0)
        with self.assertRaises(ValueError):
            DecisionPolicy(overhead_ms=math.inf)


class JevDeciderTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_independent_noul_checks_and_preserves_probabilities(self):
        calls = []

        async def transport(payload):
            calls.append(payload)
            return response_for(payload, (0.991, 0.992, 0.80, 0.999))

        decider = JevDecider(transport=transport)
        result = await decider.decide(
            {"query": "Explain the return policy."}, [candidate(), candidate("b")], "answer"
        )
        self.assertEqual(result.candidate_id, "entry-a")
        self.assertEqual(result.model, "jev-1.13.0")
        self.assertEqual(result.probabilities["entry-a:equivalent"], 0.991)
        self.assertEqual(result.probabilities["b:satisfies"], 0.999)
        self.assertGreaterEqual(result.elapsed_ms, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["questions"]), 4)
        self.assertEqual(calls[0]["state"]["candidates"][0]["artifact"], candidate().artifact)
        self.assertTrue(all(q["type"] == "noul" for q in calls[0]["questions"].values()))

    async def test_both_checks_must_pass_without_probability_multiplication(self):
        async def transport(payload):
            return response_for(payload, (0.99, 0.99))

        result = await JevDecider(transport=transport).decide({}, [candidate()], "answer")
        self.assertEqual(result.candidate_id, "entry-a")

        async def rejected(payload):
            return response_for(payload, (1.0, 0.98))

        result = await JevDecider(transport=rejected).decide({}, [candidate()], "answer")
        self.assertIsNone(result.candidate_id)
        self.assertEqual(len(result.probabilities), 2)

    async def test_conservative_ranking_and_stable_ties(self):
        async def transport(payload):
            return response_for(payload, (0.999, 0.991, 0.995, 0.995))

        result = await JevDecider(transport=transport).decide(
            {}, [candidate("a"), candidate("b")], "answer"
        )
        self.assertEqual(result.candidate_id, "b")

        async def tied(payload):
            return response_for(payload)

        result = await JevDecider(transport=tied).decide(
            {}, [candidate("z"), candidate("a")], "answer"
        )
        self.assertEqual(result.candidate_id, "a")

    async def test_plan_and_intermediate_questions_are_specific_and_bounded(self):
        for kind, expected in (
            ("plan", "tools will execute again"),
            ("intermediate", "intermediate output"),
        ):
            calls = []

            async def transport(payload):
                calls.append(payload)
                return response_for(payload)

            await JevDecider(transport=transport).decide({}, [candidate()], kind)
            questions = list(calls[0]["questions"].values())
            text = " ".join(q["instructions"]["question"] for q in questions)
            self.assertIn(expected, text)
            self.assertIn("Do not generate", questions[0]["instructions"]["rules"])

    async def test_empty_candidates_make_no_provider_call(self):
        async def transport(payload):
            self.fail("empty retrieval must not call Jev")

        result = await JevDecider(transport=transport).decide({}, [], "answer")
        self.assertIsNone(result.candidate_id)
        self.assertEqual(result.elapsed_ms, 0)

    async def test_malformed_provider_outputs_fail_closed(self):
        mutations = (
            lambda body: body.update(model="jev-latest"),
            lambda body: body.pop("model"),
            lambda body: body["answers"].pop("c0_equivalent"),
            lambda body: body["answers"].update(extra={"type": "noul", "noul": 1}),
            lambda body: body["answers"]["c0_equivalent"].update(type="score"),
        )
        for mutate in mutations:

            async def transport(payload):
                body = response_for(payload)
                mutate(body)
                return body

            with self.subTest(mutation=mutate), self.assertRaises(DecisionError):
                await JevDecider(transport=transport).decide({}, [candidate()], "answer")

        for invalid in (-0.1, 1.1, math.nan, math.inf, True, "0.99", None, 10**1000):

            async def transport(payload):
                return response_for(payload, (invalid, 1.0))

            with self.subTest(probability=invalid), self.assertRaises(DecisionError):
                await JevDecider(transport=transport).decide({}, [candidate()], "answer")

    async def test_timeout_is_bounded_without_retry(self):
        calls = []

        async def transport(payload):
            calls.append(payload)
            await asyncio.sleep(1)
            return response_for(payload)

        with self.assertRaisesRegex(DecisionError, "timed out"):
            await JevDecider(transport=transport, timeout_s=0.01).decide(
                {}, [candidate()], "answer"
            )
        self.assertEqual(len(calls), 1)

    async def test_transport_error_does_not_expose_provider_details(self):
        async def transport(payload):
            raise RuntimeError("secret-api-key response body")

        with self.assertRaises(DecisionError) as caught:
            await JevDecider(transport=transport).decide({}, [candidate()], "answer")
        self.assertNotIn("secret-api-key", str(caught.exception))

    async def test_missing_credentials_fail_without_network(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("urllib.request.build_opener") as opener,
        ):
            with self.assertRaisesRegex(DecisionError, "TYPESAFE_API_KEY"):
                await JevDecider().decide({}, [candidate()], "answer")
            opener.assert_not_called()

    async def test_invalid_candidate_context_fails_before_transport(self):
        async def transport(payload):
            self.fail("invalid context must not call transport")

        cases = (
            [candidate(), candidate()],
            [candidate("")],
            [candidate(artifact=object())],
            [candidate(artifact=math.nan)],
            [candidate(avoided_ms=-1)],
        )
        for candidates in cases:
            with self.subTest(candidates=candidates), self.assertRaises(DecisionError):
                await JevDecider(transport=transport).decide({}, candidates, "answer")
        with self.assertRaises(DecisionError):
            await JevDecider(transport=transport, max_request_bytes=10).decide(
                {}, [candidate()], "answer"
            )
        with self.assertRaises(DecisionError):
            await JevDecider(transport=transport, max_candidates=1).decide(
                {}, [candidate("a"), candidate("b")], "answer"
            )

    async def test_stdlib_transport_uses_documented_endpoint_and_bounded_timeout(self):
        provider_response = MagicMock()
        provider_response.__enter__.return_value = provider_response
        provider_response.read.side_effect = lambda limit: json.dumps(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "c0_equivalent": {"type": "noul", "noul": 0.999},
                    "c0_satisfies": {"type": "noul", "noul": 0.999},
                },
            }
        ).encode()
        opener = MagicMock()
        opener.open.return_value = provider_response
        with patch("urllib.request.build_opener", return_value=opener):
            result = await JevDecider(api_key="fake-key", timeout_s=0.5).decide(
                {}, [candidate()], "answer"
            )
        self.assertEqual(result.candidate_id, "entry-a")
        self.assertEqual(opener.open.call_count, 1)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-key")
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 0.5)

    async def test_provider_http_failures_do_not_retry(self):
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError(
            "https://api.typesafe.ai/v1/systemone", 429, "secret provider detail", {}, None
        )
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(DecisionError, "HTTP 429"):
                await JevDecider(api_key="fake-key").decide({}, [candidate()], "answer")
        self.assertEqual(opener.open.call_count, 1)

    async def test_ambiguous_and_nonfinite_json_fails_closed(self):
        for raw in (
            b'{"model":"jev-1.13.0", "model":"jev-1.13.0", "answers":{}}',
            b'{"model":"jev-1.13.0", "answers":{"x":{"noul":NaN}}}',
            b"invalid json",
        ):
            provider_response = MagicMock()
            provider_response.__enter__.return_value = provider_response
            provider_response.read.return_value = raw
            opener = MagicMock()
            opener.open.return_value = provider_response
            with patch("urllib.request.build_opener", return_value=opener):
                with self.subTest(raw=raw), self.assertRaises(DecisionError):
                    await JevDecider(api_key="fake-key").decide({}, [candidate()], "answer")


if __name__ == "__main__":
    unittest.main()
