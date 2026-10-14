"""Action-choice analysis checks."""

import unittest

from causal_interventions.analysis import EpisodeResult, EpisodeTruth
from causal_interventions.analysis_v2 import analyze_v2


class AnalysisV2Tests(unittest.TestCase):
    def test_primary_uses_action_bayes_and_informative_only(self) -> None:
        references = [EpisodeTruth(str(i), str(i), 0, (0.8,)) for i in range(3)]
        rows = [EpisodeResult(e.episode_id, e.parameter_id, policy, 0,
                              (forecast,), e.query_truths)
                for e in references
                for policy, forecast in (("model", 0.5), ("model:action_bayes", 0.8),
                                         ("random", 0.5), ("exact", 0.8))]
        result = analyze_v2(references, rows, "model", {"0", "1"}, bootstrap_samples=20)
        self.assertEqual(result["primary_action_choice"]["episodes"], 2)
        self.assertLess(result["primary_action_choice"]["paired_model_minus_random"], 0)
        self.assertEqual(result["secondary_submitted_forecast"]["episodes"], 3)
        self.assertEqual(result["secondary_submitted_forecast"]["paired_model_minus_random"], 0)
        with self.assertRaisesRegex(ValueError, "must be complete"):
            analyze_v2(references, rows[:-3], "model", {"0", "1"})
