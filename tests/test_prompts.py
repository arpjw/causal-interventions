"""Prompt stability, public information boundary, and budget/history semantics."""

from dataclasses import replace
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from causal_interventions.episodes import GeneratorConfig, Stratum, generate_split
from causal_interventions.prompts import HistoryStep, render_prompt, template_text
from causal_interventions.types import Intervention


def payload(text: str) -> dict:
    return json.loads(text.split("PUBLIC_TASK_JSON\n", 1)[1])


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.episode = generate_split(GeneratorConfig(strata=(Stratum("prompt_dev"),)),
                                     "development", 1)[0][0]

    def test_public_information_and_action_schema(self) -> None:
        text = render_prompt(self.episode, 1)
        self.assertEqual(text, render_prompt(self.episode, 1))
        data = payload(text)
        self.assertEqual(data["joint"], list(self.episode.worlds[0].joint))
        self.assertEqual(data["prior"], list(self.episode.prior))
        self.assertEqual([c["order"] for c in data["candidates"]],
                         [list(w.order) for w in self.episode.worlds])
        schema = data["response_schema"]
        self.assertEqual(schema["required"], ["action_id"])
        self.assertIn("STOP", schema["properties"]["action_id"]["enum"])
        self.assertFalse(schema["additionalProperties"])

    def test_private_state_cannot_change_prompt(self) -> None:
        changed = replace(self.episode, hidden_world=(self.episode.hidden_world + 1) % 3,
                          generator_seed=111, outcome_seed=222, episode_id="private_id",
                          parameter_id="private_parameters", split="private_split",
                          diagnostics={"private": "do not render"})
        self.assertEqual(render_prompt(self.episode, 1), render_prompt(changed, 1))
        data = payload(render_prompt(self.episode, 1))
        for key in ("hidden_world", "generator_seed", "outcome_seed", "episode_id",
                    "parameters", "parameter_id", "split", "diagnostics"):
            self.assertNotIn(key, data)

    def test_observed_and_invalid_history_forecasts(self) -> None:
        history = (HistoryStep(Intervention("X", 0), (0, 1, 1)),)
        data = payload(render_prompt(self.episode, 0, history, "forecast"))
        self.assertEqual(data["history"][0]["outcome"], [0, 1, 1])
        self.assertEqual(data["response_schema"]["properties"]["forecasts"]["required"],
                         ["P(Z=1|do(X=1))"])
        data = payload(render_prompt(self.episode, 0, (HistoryStep(None, None),), "forecast"))
        self.assertEqual(data["history"][0]["status"], "invalid")
        # Early STOP triggers a forecast with remaining budget unchanged.
        self.assertEqual(payload(render_prompt(self.episode, 1, (), "forecast"))["remaining_budget"], 1)

    def test_rejects_inconsistent_history(self) -> None:
        for history in ((HistoryStep(Intervention("X", 1), (1, 0, 0)),),
                        (HistoryStep(Intervention("X", 0), (1, 0, 0)),),
                        (HistoryStep(Intervention("X", 0), (0, 0, True)),),
                        (HistoryStep(None, (0, 0, 0)),)):
            with self.assertRaises(ValueError):
                render_prompt(self.episode, 0, history, "forecast")
        for budget in (-1, True, 2):
            with self.assertRaises(ValueError):
                render_prompt(self.episode, budget)
        with self.assertRaises(ValueError):
            render_prompt(self.episode, 0)

    def test_changed_template_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.txt"
            path.write_text("changed", encoding="utf-8")
            with patch("causal_interventions.prompts.TEMPLATE_PATH", path):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    template_text()


if __name__ == "__main__":
    unittest.main()
