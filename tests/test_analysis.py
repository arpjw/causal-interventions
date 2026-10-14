"""Registered pairing, clustering, and failure handling for analysis."""
from dataclasses import replace
import math
import unittest

from causal_interventions.analysis import EpisodeResult, EpisodeTruth, analyze_paired
from causal_interventions.scoring import bernoulli_kl


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.reference = tuple(EpisodeTruth(str(i), str(i), 0, (0.8,)) for i in range(3))
        self.results = tuple(EpisodeResult(e.episode_id, e.parameter_id, policy, 0,
                                          (prediction,), e.query_truths)
                             for e in self.reference
                             for policy, prediction in (("model", 0.7), ("random", 0.5), ("exact", 0.8)))

    def test_hand_paired_difference_and_repeatability(self):
        result = analyze_paired(self.reference, self.results, "model", bootstrap_samples=100)
        expected = bernoulli_kl(0.8, 0.7) - bernoulli_kl(0.8, 0.5)
        self.assertAlmostEqual(result["paired_model_minus_random"], expected)
        self.assertAlmostEqual(result["gap_closure"], -expected / bernoulli_kl(0.8, 0.5))
        self.assertEqual(result, analyze_paired(self.reference, tuple(reversed(self.results)), "model", bootstrap_samples=100))
        self.assertEqual(result["paired_difference_interval_95"], [expected, expected])

    def test_missing_and_invalid_model_rows_retained(self):
        rows = tuple(replace(r, submitted_probabilities=(math.nan,), parse_failures=1, cost=0.2)
                     if r.policy_id == "model" else r
                     for r in self.results if not (r.policy_id == "model" and r.episode_id == "0"))
        result = analyze_paired(self.reference, rows, "model", bootstrap_samples=20)
        self.assertEqual(result["episodes"], 3)
        self.assertEqual(len(result["episode_scores"]), 9)
        self.assertEqual(result["counts"]["model"]["missing_rows"], 1)
        self.assertEqual(result["counts"]["model"]["invalid_forecasts"], 3)
        self.assertEqual(result["counts"]["model"]["malformed_rows"], 2)
        self.assertEqual(result["counts"]["model"]["parse_failures"], 2)
        self.assertAlmostEqual(result["counts"]["model"]["cost"], 0.4)
        self.assertEqual(result["paired_model_minus_random"], 0)

    def test_baseline_failure_is_protocol_error(self):
        rows = tuple(r for r in self.results if not (r.policy_id == "random" and r.episode_id == "0"))
        with self.assertRaisesRegex(ValueError, "protocol failure"):
            analyze_paired(self.reference, rows, "model")
        rows = tuple(replace(r, submitted_probabilities=(None,)) if r.policy_id == "exact" else r for r in self.results)
        with self.assertRaisesRegex(ValueError, "protocol failure"):
            analyze_paired(self.reference, rows, "model")

    def test_truth_corruption_and_duplicate_rejected(self):
        for field in ({"true_world": 1}, {"query_truths": (0.7,)}, {"parameter_id": "wrong"}):
            with self.assertRaisesRegex(ValueError, "disagree"):
                analyze_paired(self.reference, (replace(self.results[0], **field),) + self.results[1:], "model")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_paired(self.reference, self.results + (self.results[0],), "model")

    def test_cluster_weighting_prevents_replication_inflation(self):
        reference = self.reference + (EpisodeTruth("repeat", "0", 0, (0.8,)),)
        varied = tuple(replace(r, submitted_probabilities=(0.8,))
                       if r.episode_id == "0" and r.policy_id == "model" else r
                       for r in self.results)
        rows = varied + tuple(replace(r, episode_id="repeat") for r in varied if r.episode_id == "0")
        result = analyze_paired(reference, rows, "model", bootstrap_samples=20)
        original = analyze_paired(self.reference, varied, "model", bootstrap_samples=20)
        self.assertEqual(result["independent_parameter_clusters"], 3)
        self.assertEqual(result["paired_model_minus_random"], original["paired_model_minus_random"])
        self.assertEqual(result["paired_difference_interval_95"], original["paired_difference_interval_95"])

    def test_small_gap_suppresses_closure_single_cluster_no_interval(self):
        result = analyze_paired(self.reference[:1], self.results[:3], "model", minimum_gap=1)
        self.assertIsNone(result["gap_closure"])
        self.assertIsNone(result["paired_difference_interval_95"])


if __name__ == "__main__":
    unittest.main()
