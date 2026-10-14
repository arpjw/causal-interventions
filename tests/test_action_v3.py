"""Expected-risk scoring and public-input controls for the action-only study."""

import unittest

from causal_interventions.action_v3 import (
    action_values, parse_action, render_action_prompt, run_action, summarize_action_runs,
)
from causal_interventions.episodes import GeneratorConfig, Stratum, generate_split
from causal_interventions.runner import Generation


class FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        self.prompts.append(prompt)
        return Generation(self.response, 10, 5, 0.1)


class ActionV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = GeneratorConfig(
            seed=90210, analysis_version=3, coefficient_bound=1.2, min_cell=0.0002,
            strata=(Stratum("informative", candidate_count=2, min_query_range=0.08,
                            min_action_gap=0.01, min_opportunity_gap=0.01),),
        )
        cls.episodes, _ = generate_split(config, "development", 6)

    def test_balanced_optimal_labels(self) -> None:
        winners = [min(action_values(episode), key=action_values(episode).get)
                   for episode in self.episodes]
        self.assertEqual(sorted(winners.count(choice) for choice in action_values(self.episodes[0])),
                         [2, 2, 2])

    def test_prompt_uses_only_public_values(self) -> None:
        episode = self.episodes[0]
        base = render_action_prompt(episode)
        assisted = render_action_prompt(episode, assisted=True)
        for text in (base, assisted):
            self.assertNotIn(episode.episode_id, text)
            self.assertNotIn(episode.parameter_id, text)
            self.assertNotIn("hidden_world", text)
            self.assertNotIn("outcome_seed", text)
        self.assertNotIn('"computed_action_values":', base)
        self.assertIn('"computed_action_values":', assisted)

    def test_exact_action_and_invalid_fallback(self) -> None:
        episode = self.episodes[0]
        best = min(action_values(episode), key=action_values(episode).get)
        backend = FakeBackend('{"action_id":"' + best + '"}')
        row = run_action(episode, backend, assisted=False)
        self.assertEqual(len(backend.prompts), 1)
        self.assertFalse(row["invalid_action"])
        self.assertAlmostEqual(row["model_risk"], row["oracle_risk"])
        invalid = run_action(episode, FakeBackend('{"action_id":"do(X=1)"}'), assisted=False)
        self.assertTrue(invalid["invalid_action"])
        self.assertIsNone(invalid["selected_action"])
        self.assertAlmostEqual(invalid["model_risk"], invalid["initial_risk"])
        self.assertEqual(parse_action('oops', episode), (None, True))

    def test_replay_and_uncertainty(self) -> None:
        runs = []
        for episode in self.episodes:
            best = min(action_values(episode), key=action_values(episode).get)
            runs.append(run_action(episode, FakeBackend('{"action_id":"' + best + '"}'), False))
        report = summarize_action_runs(self.episodes, runs, bootstrap_samples=100)
        self.assertLess(report["model_minus_random"], 0)
        self.assertAlmostEqual(report["mean_risks"]["model_risk"],
                               report["mean_risks"]["oracle_risk"])
        runs[0]["model_risk"] += 0.01
        with self.assertRaisesRegex(ValueError, "replay"):
            summarize_action_runs(self.episodes, runs)


if __name__ == "__main__":
    unittest.main()
