"""Action aliasing, information-ladder integrity, and score replay."""

import json
import unittest

from causal_interventions.action_v4 import (action_labels, parse_action, public_task,
                                             render_action_prompt, run_action)
from causal_interventions.action_v3 import action_values
from causal_interventions.episodes import GeneratorConfig, Stratum, generate_split
from causal_interventions.runner import Generation


class FakeBackend:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        return Generation(self.response, 10, 5, 0.1)


class ActionV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = GeneratorConfig(
            seed=20261013, analysis_version=4, coefficient_bound=1.2, min_cell=0.0002,
            strata=(Stratum("informative", candidate_count=2, min_query_range=0.08,
                            min_action_gap=0.01, min_opportunity_gap=0.01),),
        )
        cls.episodes, _ = generate_split(config, "development", 6)

    def test_balanced_optimal_actions_and_opaque_permutation(self) -> None:
        winners = [min(action_values(e), key=action_values(e).get) for e in self.episodes]
        self.assertEqual(sorted(winners.count(a) for a in action_values(self.episodes[0])),
                         [2, 2, 2])
        for episode in self.episodes:
            labels = action_labels(episode, "opaque_raw")
            self.assertEqual(set(labels), {"A", "B", "C"})
            self.assertEqual(set(labels.values()), set(action_values(episode)))
            self.assertEqual(labels, action_labels(episode, "opaque_values"))

    def test_information_ladder_has_no_hidden_state(self) -> None:
        episode = self.episodes[0]
        for condition in ("literal_raw", "opaque_raw", "opaque_likelihood",
                          "opaque_values", "opaque_thinking"):
            text = render_action_prompt(episode, condition)
            task = public_task(episode, condition)
            self.assertEqual(json.loads(text.split("PUBLIC_TASK_JSON\n", 1)[1]), task)
            self.assertNotIn(episode.episode_id, text)
            self.assertNotIn(episode.parameter_id, text)
            self.assertNotIn("hidden_world", text)
            self.assertNotIn("outcome_seed", text)
        self.assertNotIn("intermediate_distributions", public_task(episode, "opaque_raw"))
        self.assertNotIn("computed_action_values", public_task(episode, "opaque_raw"))
        table = public_task(episode, "opaque_likelihood")["intermediate_distributions"]
        self.assertEqual(len(table["outcome_states_lexicographic"]), 8)
        for by_world in table["experiment_outcome_probability_by_action_and_candidate"].values():
            for probabilities in by_world.values():
                self.assertAlmostEqual(sum(probabilities), 1)
        self.assertNotIn("computed_action_values", public_task(episode, "opaque_likelihood"))

    def test_opaque_response_maps_back_to_scored_action(self) -> None:
        episode = self.episodes[0]
        labels = action_labels(episode, "opaque_raw")
        best = min(action_values(episode), key=action_values(episode).get)
        label = next(label for label, semantic in labels.items() if semantic == best)
        row = run_action(episode, FakeBackend(json.dumps({"action_id": label})),
                         "opaque_raw", 96)
        self.assertEqual(row["selected_label"], label)
        self.assertEqual(row["selected_action"], best)
        self.assertAlmostEqual(row["model_risk"], row["oracle_risk"])
        self.assertEqual(parse_action('{"action_id":"Z"}', episode, "opaque_raw"),
                         (None, None, True))


if __name__ == "__main__":
    unittest.main()
