"""Prompt changes must isolate calibration and preserve original native evidence."""

import json
import unittest
from pathlib import Path

from test_laya import Agent

from jac import LayaDecider
from jevcache.decisions import Candidate
from jevcache.laya_prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    laya_approval_values,
    laya_questions,
)


class PromptTests(unittest.TestCase):
    def test_v2_questions_match_frozen_contract(self):
        source = (Path(__file__).resolve().parent
                  / "fixtures/reuse_questions.json")
        values = json.loads(source.read_text(encoding="utf-8"))
        for kind, questions in values["_QUESTIONS"].items():
            expected = {
                name: {"type": "noul", "instructions": question + values["_RULES"],
                       "criteria": values["_CRITERIA"]}
                for name, question in zip(("equivalent", "satisfies"), questions, strict=True)
            }
            self.assertEqual(laya_questions(kind), expected)

    def test_versions_isolate_calibration_identity_and_keep_v2_default(self):
        deciders = [LayaDecider(agent=Agent(), prompt_version=version)
                    for version in PROMPT_VERSIONS]
        try:
            identities = [decider.cache_identity for decider in deciders]
            self.assertEqual(len({item["question_schema_sha256"] for item in identities}),
                             len(PROMPT_VERSIONS))
            self.assertEqual([item["question_schema_version"] for item in identities],
                             list(PROMPT_VERSIONS))
            with self.assertRaises(ValueError):
                LayaDecider(prompt_version="not-a-version")
            self.assertEqual(deciders[0].prompt_version, DEFAULT_PROMPT_VERSION)
        finally:
            for decider in deciders:
                decider.close()

    def test_question_results_do_not_share_mutable_criteria(self):
        questions = laya_questions(version=PROMPT_VERSIONS[-1])
        questions["equivalent"]["criteria"]["true"] = "changed"
        self.assertNotEqual(laya_questions(version=PROMPT_VERSIONS[-1]), questions)
        with self.assertRaises(ValueError):
            laya_questions("unknown")


class NativePromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_conflict_questions_invert_native_scores_only_for_answers(self):
        from test_laya import response

        version = "laya-reuse-v3-conflicts"
        for kind, native in (("answer", (.01, .09)), ("plan", (.99, .99))):
            agent = Agent([response(native)])
            decider = LayaDecider(agent=agent, prompt_version=version, threshold=.9)
            try:
                result = await decider.decide({}, [Candidate("a", {}, "answer", 500)], kind)
                self.assertEqual(result.candidate_id, "a")
                self.assertGreaterEqual(min(result.probabilities.values()), .9)
                self.assertEqual(laya_approval_values(kind, version)["equivalent"],
                                 kind != "answer")
            finally:
                await decider.drain()
                decider.close()

    async def test_new_profiles_preserve_full_evidence_and_two_check_rule(self):
        from test_laya import response

        for version in PROMPT_VERSIONS[1:]:
            native = response((.99, .89))
            for name in laya_questions(version=version):
                native["answers"].setdefault(name, {"type": "noul", "noul": .99})
            agent = Agent([native])
            decider = LayaDecider(agent=agent, prompt_version=version, threshold=.9)
            new = {"query": "Generate required code.", "history": ["Original context"]}
            stored = Candidate("a", {"query": "Generate required code."},
                               {"content": "Incomplete explanation"}, 500)
            try:
                result = await decider.decide(new, [stored])
                self.assertIsNone(result.candidate_id)
                state, questions, _ = agent.calls[0]
                self.assertEqual(json.loads(state), {"new_request": new,
                                                    "cached_request": stored.request,
                                                    "artifact": stored.artifact})
                self.assertEqual(questions, laya_questions(version=version))
            finally:
                await decider.drain()
                decider.close()


if __name__ == "__main__":
    unittest.main()
