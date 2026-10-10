"""Prompt changes must isolate calibration and preserve original native evidence."""

import ast
import json
import unittest
from pathlib import Path

from test_laya import Agent

from jac import LayaDecider
from jevcache.decisions import Candidate
from jevcache.laya_prompt_benchmark import summarize
from jevcache.laya_prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    laya_approval_values,
    laya_questions,
)


class PromptTests(unittest.TestCase):
    def test_v2_questions_match_archived_adapter_exactly(self):
        source = (Path(__file__).resolve().parents[1]
                  / "tests/fixtures/laya_v2_adapter.py.txt")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name)
                            and target.id in ("_QUESTIONS", "_RULES", "_CRITERIA")):
                        values[target.id] = ast.literal_eval(node.value)
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

    def test_quality_summary_excludes_exact_and_records_missed_positives(self):
        def row(identifier, label, expected, accepted, score):
            return {"id": identifier, "label": label, "expected_id": expected,
                    "candidate_id": "a", "accepted": accepted,
                    "correct": ("a" if accepted else None) == expected, "score": score,
                    "native_predict_calls": 1, "context_rejections": 0,
                    "error": None, "elapsed_ms": 10}
        result = summarize([
            row("exact", "exact", "a", True, .99),
            row("true", "positive", "a", True, .99),
            row("false", "hard_negative", None, True, .99),
            row("missed", "positive", "a", False, .7),
        ])
        self.assertEqual(result["cases"], 4)
        self.assertEqual(result["semantic_cases"], 3)
        self.assertEqual(result["true_semantic_accepts"], 1)
        self.assertEqual(result["false_semantic_accepts"], 1)
        self.assertEqual(result["semantic_precision"], .5)
        self.assertEqual(result["positive_semantic_recall"], .5)
        self.assertEqual(result["missed_positive_ids"], ["missed"])


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
