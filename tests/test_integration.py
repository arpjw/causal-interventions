"""Independent checks that generated episodes and decision policies agree."""

import unittest

from causal_interventions.beliefs import BeliefModel
from causal_interventions.episodes import GeneratorConfig, Stratum, generate_split
from causal_interventions.policies import ExactPolicy, RandomPolicy, expected_policy_risk


class IntegrationTests(unittest.TestCase):
    def test_generator_diagnostics_match_policy_engine(self) -> None:
        config = GeneratorConfig(
            strata=(Stratum("audit", candidate_count=3, budget=1),)
        )
        episodes, _ = generate_split(config, "development", 8)
        for episode in episodes:
            with self.subTest(episode=episode.episode_id):
                model = BeliefModel(
                    episode.worlds, episode.actions, episode.queries, episode.prior
                )
                exact = ExactPolicy(model)
                random = RandomPolicy(model, seed=123)
                self.assertAlmostEqual(
                    model.risk(model.prior),
                    episode.diagnostics["initial_risk"],
                    places=10,
                )
                self.assertAlmostEqual(
                    exact.value(model.prior, 1),
                    episode.diagnostics["one_step_oracle_risk"],
                    places=10,
                )
                self.assertAlmostEqual(
                    expected_policy_risk(model, random, model.prior, 1),
                    episode.diagnostics["one_step_random_risk"],
                    places=10,
                )
                self.assertLessEqual(exact.value(model.prior, 1), model.risk(model.prior))


if __name__ == "__main__":
    unittest.main()
