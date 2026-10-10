import unittest

from jevcache.realistic_benchmark import guard, summarize, thresholds
from jevcache.realistic_workload import FAMILIES, requests


class RealisticBenchmarkTests(unittest.TestCase):
    def test_labels_and_guards(self):
        rows = requests()
        canonicals = {family[0]: family[2] for family in FAMILIES}
        self.assertEqual(len(canonicals), 20)
        self.assertEqual(len(rows), 88)
        self.assertEqual(len({row["id"] for row in rows}), 88)
        self.assertEqual(sum(row["label"] == "positive" for row in rows), 40)
        for row in rows:
            if row["label"] == "positive":
                self.assertIn(row["expected_id"], canonicals)
                self.assertNotEqual(row["text"], canonicals[row["expected_id"]])
                self.assertEqual(guard(row["text"]), guard(canonicals[row["expected_id"]]))
            else:
                self.assertIsNone(row["expected_id"])
        for key, _, canonical, _, negatives in FAMILIES:
            if key in ("percentage", "portfolio"):
                self.assertTrue(any(guard(text) != guard(canonical) for text in negatives))

    def test_metrics_do_not_treat_a_candidate_as_a_correct_reuse(self):
        rows = [
            {"id": "p1", "label": "positive", "expected_id": "a",
             "candidates": ["a", "b"], "top_cosine": .96, "retrieval_ms": [10, 20]},
            {"id": "p2", "label": "positive", "expected_id": "b",
             "candidates": ["a", "b"], "top_cosine": .94, "retrieval_ms": [30, 40]},
            {"id": "n", "label": "hard_negative", "expected_id": None,
             "candidates": ["a"], "top_cosine": .97, "retrieval_ms": [50, 60]},
        ]
        metrics = summarize(rows)
        self.assertEqual(metrics["positive_recall_at_1"], .5)
        self.assertEqual(metrics["positive_recall_at_5"], 1)
        self.assertEqual(metrics["p50_ms"], 35)
        self.assertEqual(metrics["p95_ms"], 60)
        self.assertEqual(metrics["negatives_with_candidates"], 1)
        diagnostic = thresholds(rows)[-1]
        self.assertEqual(diagnostic["correct_accepts"], 1)
        self.assertEqual(diagnostic["incorrect_accepts"], 1)
        self.assertEqual(diagnostic["precision"], .5)
