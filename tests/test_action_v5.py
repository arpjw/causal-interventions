"""V5 public-input stage truth, strict scoring, and calculator transcript tests."""

from __future__ import annotations

import json
import unittest

from causal_interventions.action_v5 import (action_labels, public_task, render_stage_prompt,
                                            render_tool_prompt, run_tool_agent, score_stage,
                                            stage_schema, stage_truth)
from causal_interventions.episodes import GeneratorConfig, Stratum, generate_split
from causal_interventions.runner import Generation


class SequenceBackend:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        if not self.responses:
            raise AssertionError("unexpected extra model call")
        return Generation(json.dumps(self.responses.pop(0)), 20, 10, 0.1)


class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = GeneratorConfig(
            seed=827420, analysis_version=4, coefficient_bound=1.2, min_cell=0.0002,
            strata=(Stratum("informative_c2", candidate_count=2, min_query_range=0.08,
                            min_action_gap=0.01, min_opportunity_gap=0.01),
                    Stratum("informative_c3", candidate_count=3, min_query_range=0.08,
                            min_action_gap=0.01, min_opportunity_gap=0.01)),
        )
        cls.episodes, _ = generate_split(config, "development", 6)

    def test_stage_truth_and_schema_for_both_world_counts(self) -> None:
        for episode in (self.episodes[0], self.episodes[-1]):
            truth = stage_truth(episode)
            self.assertEqual(set(truth["query_probabilities"]),
                             {f"W{i}" for i in range(len(episode.worlds))})
            self.assertAlmostEqual(sum(truth["posterior_probabilities"].values()), 1)
            self.assertEqual(set(truth["action_risks"]), set("ABC"))
            for condition in ("q32_raw_stages", "q235_short_raw_stages",
                              "q235_long_raw_stages", "q235_likelihood_stages"):
                task = public_task(episode, likelihoods=condition == "q235_likelihood_stages")
                self.assertNotIn("hidden_world", json.dumps(task))
                prompt = render_stage_prompt(episode, condition)
                self.assertEqual(json.loads(prompt.split("PUBLIC_TASK_JSON\n", 1)[1])["response_schema"],
                                 stage_schema(episode, condition))

    def test_exact_stage_response_scores_zero_error(self) -> None:
        episode = self.episodes[0]
        truth = stage_truth(episode)
        label = min(truth["action_risks"], key=truth["action_risks"].get)
        for condition in ("q32_raw_stages", "q235_likelihood_stages"):
            schema = stage_schema(episode, condition)
            response = {key: truth[key] if key != "action_id" else label
                        for key in schema["required"]}
            row = score_stage(episode, condition, Generation(json.dumps(response), 20, 10, 0.1))
            self.assertEqual(row["selected_label"], label)
            self.assertTrue(row["self_consistent"])
            self.assertAlmostEqual(row["model_risk"], row["oracle_risk"])
            self.assertTrue(all(value == 0 for value in row["stage_mae"].values()))

    def test_invalid_numeric_stage_does_not_drop_episode(self) -> None:
        episode = self.episodes[0]
        truth = stage_truth(episode)
        response = {name: truth[name] for name in ("query_probabilities", "probe_likelihoods",
                                                  "posterior_probabilities", "action_risks")}
        response["query_probabilities"] = {key: -1 for key in truth["query_probabilities"]}
        response["action_id"] = "STOP"
        row = score_stage(episode, "q32_raw_stages", Generation(json.dumps(response), 20, 10, 0.1))
        self.assertIsNone(row["stage_mae"]["query_probabilities"])
        self.assertEqual(row["model_risk"], row["initial_risk"])
        self.assertFalse(row["invalid_action"])

    def test_tool_agent_can_query_all_actions_and_choose_minimum(self) -> None:
        episode = self.episodes[0]
        truth = stage_truth(episode)
        label = min(truth["action_risks"], key=truth["action_risks"].get)
        backend = SequenceBackend([{"operation": "evaluate", "action_id": key} for key in "ABC"]
                                  + [{"operation": "final", "action_id": label}])
        row = run_tool_agent(episode, backend)
        self.assertEqual(row["distinct_tool_actions"], 3)
        self.assertEqual(row["selected_action"], action_labels(episode)[label])
        self.assertAlmostEqual(row["model_risk"], row["oracle_risk"])
        self.assertEqual(len(row["calls"]), 4)
        self.assertIn("You have 0 evaluation calls remaining",
                      render_tool_prompt(episode, row["tool_transcript"]))


if __name__ == "__main__":
    unittest.main()
