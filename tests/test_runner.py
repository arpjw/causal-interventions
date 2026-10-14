"""Development runner protocol tests using a deterministic fake backend."""

from __future__ import annotations

import json
import unittest

from causal_interventions.episodes import GeneratorConfig, Stratum, generate_split
from causal_interventions.prompts import action_id, query_id
from causal_interventions.runner import Generation, run_episode
from causal_interventions.beliefs import BeliefModel
from causal_interventions.factorize import assignment_index


class FakeBackend:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        self.prompts.append(prompt)
        return Generation(next(self.responses), 100, 10, 0.1)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = GeneratorConfig(strata=(Stratum("informative"),))
        cls.episode = generate_split(config, "development", 1)[0][0]
        cls.test_episode = generate_split(config, "test", 1)[0][0]

    def test_valid_action_and_forecast(self) -> None:
        episode = self.episode
        backend = FakeBackend([
            json.dumps({"action_id": action_id(episode.actions[0])}),
            json.dumps({"forecasts": {query_id(q): 0.6 for q in episode.queries}}),
        ])
        run = run_episode(episode, backend, "fake")
        self.assertEqual([r.policy_id for r in run.results], ["fake", "random", "exact"])
        self.assertEqual(run.results[0].submitted_probabilities, (0.6,))
        self.assertEqual(run.trace["parse_failures"], 0)
        self.assertEqual(len(run.trace["history"]), 1)
        self.assertEqual(run.trace["input_tokens"], 200)
        for prompt in backend.prompts:
            self.assertNotIn(episode.episode_id, prompt)
            self.assertNotIn(str(episode.outcome_seed), prompt)

    def test_action_score_uses_observed_history_not_submitted_forecast(self) -> None:
        episode = self.episode
        backend = FakeBackend([
            json.dumps({"action_id": action_id(episode.actions[0])}),
            json.dumps({"forecasts": {query_id(q): 0.5 for q in episode.queries}}),
        ])
        run = run_episode(episode, backend, "fake", score_actions=True,
                          prompt_version="primary-v2")
        model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
        observed = tuple(run.trace["history"][0]["outcome"])
        expected = model.predictive(model.posterior(
            model.prior, episode.actions[0], assignment_index(observed)))
        self.assertEqual(run.results[-1].policy_id, "fake:action_bayes")
        self.assertEqual(run.results[-1].submitted_probabilities, expected)
        self.assertIn('"template_version":"primary-v2"', backend.prompts[0])

    def test_invalid_action_has_prior_action_score(self) -> None:
        run = run_episode(self.episode, FakeBackend(["bad", "bad"]), "fake",
                          score_actions=True)
        model = BeliefModel(self.episode.worlds, self.episode.actions,
                            self.episode.queries, self.episode.prior)
        self.assertEqual(run.results[-1].submitted_probabilities, model.predictive(model.prior))

    def test_invalid_action_consumes_budget_and_forecast_defaults(self) -> None:
        backend = FakeBackend(["not json", "not json"])
        run = run_episode(self.episode, backend, "fake")
        self.assertEqual(run.results[0].submitted_probabilities, (0.5,))
        self.assertEqual(run.trace["parse_failures"], 2)
        self.assertEqual(run.trace["remaining_budget"], 0)

    def test_stop_retains_budget(self) -> None:
        backend = FakeBackend(['{"action_id":"STOP"}',
                               json.dumps({"forecasts": {query_id(q): 0.5 for q in self.episode.queries}})])
        run = run_episode(self.episode, backend, "fake")
        self.assertEqual(run.trace["remaining_budget"], self.episode.budget)
        self.assertEqual(run.trace["history"], [])

    def test_test_split_cannot_call_model(self) -> None:
        backend = FakeBackend([])
        with self.assertRaises(ValueError):
            run_episode(self.test_episode, backend, "fake")
        self.assertFalse(backend.prompts)

    def test_verified_lock_runner_can_call_test_split(self) -> None:
        backend = FakeBackend([
            '{"action_id":"STOP"}',
            json.dumps({"forecasts": {query_id(q): 0.5 for q in self.test_episode.queries}}),
        ])
        run = run_episode(self.test_episode, backend, "fake", allow_locked_test=True)
        self.assertEqual(run.reference.episode_id, self.test_episode.episode_id)
        self.assertEqual(run.trace["parse_failures"], 0)


if __name__ == "__main__":
    unittest.main()
