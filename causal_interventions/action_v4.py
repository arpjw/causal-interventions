"""Public-input presentation and information-ladder conditions for v4."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from typing import Any

from .action_v3 import action_values, summarize_action_runs, compare_action_runs
from .episodes import Episode, STATES, canonical_json, one_step_diagnostics
from .prompts import action_id
from .runner import Backend

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "action-v4.txt"
CONDITIONS = ("literal_raw", "opaque_raw", "opaque_likelihood", "opaque_values",
              "opaque_thinking")


def action_labels(episode: Episode, condition: str) -> dict[str, str]:
    """Map presented IDs to semantic actions, with a fixed per-case permutation."""
    if condition not in CONDITIONS:
        raise ValueError("unknown v4 condition")
    semantics = [action_id(action) for action in episode.actions]
    if condition == "literal_raw":
        return {semantic: semantic for semantic in semantics}
    seed = int(sha256((episode.episode_id + ":v4-opaque").encode()).hexdigest(), 16)
    Random(seed).shuffle(semantics)
    return dict(zip(("A", "B", "C"), semantics, strict=True))


def public_task(episode: Episode, condition: str) -> dict[str, Any]:
    if episode.budget != 1 or len(episode.actions) != 3 or len(episode.queries) != 1:
        raise ValueError("v4 requires one action from three choices and one query")
    public = episode.public_record()
    labels = action_labels(episode, condition)
    task: dict[str, Any] = {
        "variables": public["variables"], "joint": public["joint"],
        "candidates": [{"candidate_id": f"W{i}", "order": order}
                       for i, order in enumerate(public["orders"])],
        "prior": public["prior"],
        "allowed_actions": ([{"action_id": label, "intervention": semantic}
                             for label, semantic in labels.items()]
                            if condition != "literal_raw" else list(labels)),
        "query": public["queries"][0],
        "response_schema": {"action_id": list(labels) + ["STOP"]},
    }
    if condition == "opaque_likelihood":
        task["intermediate_distributions"] = {
            "outcome_states_lexicographic": [list(state) for state in STATES],
            "query_probability_by_candidate": {
                f"W{i}": world.probability(episode.queries[0])
                for i, world in enumerate(episode.worlds)},
            "experiment_outcome_probability_by_action_and_candidate": {
                label: {f"W{i}": list(world.intervention_distribution(
                    next(action for action in episode.actions if action_id(action) == semantic)))
                        for i, world in enumerate(episode.worlds)}
                for label, semantic in labels.items()},
        }
    if condition == "opaque_values":
        values = action_values(episode)
        task["computed_action_values"] = {
            "metric": "expected_remaining_bayes_risk_nats_lower_is_better",
            "values": {label: values[semantic] for label, semantic in labels.items()},
        }
    return task


def render_action_prompt(episode: Episode, condition: str) -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace(
        "{{PUBLIC_TASK_JSON}}", canonical_json(public_task(episode, condition)))


def parse_action(text: str, episode: Episode, condition: str) -> tuple[str | None, str | None, bool]:
    try:
        value = json.loads(text.strip())
    except (TypeError, ValueError):
        return None, None, True
    if not isinstance(value, dict) or set(value) != {"action_id"}:
        return None, None, True
    label = value["action_id"]
    labels = action_labels(episode, condition)
    if not isinstance(label, str) or label not in (*labels, "STOP"):
        return None, None, True
    return label, labels.get(label, "STOP"), False


def run_action(episode: Episode, backend: Backend, condition: str,
               max_new_tokens: int, allow_locked_test: bool = False) -> dict[str, Any]:
    if episode.split != "development" and not (episode.split == "test" and allow_locked_test):
        raise ValueError("test calls require a verified lock")
    prompt = render_action_prompt(episode, condition)
    generated = backend.generate(prompt, max_new_tokens)
    label, selected, invalid = parse_action(generated.text, episode, condition)
    values = action_values(episode)
    diagnostics = one_step_diagnostics(episode.worlds, episode.prior,
                                       episode.actions, episode.queries)
    return {
        "episode_id": episode.episode_id, "parameter_id": episode.parameter_id,
        "split": episode.split, "stratum": episode.stratum,
        "condition": condition, "prompt": prompt, "generation": asdict(generated),
        "selected_label": label, "selected_action": selected,
        "invalid_action": invalid,
        "model_risk": values.get(selected, diagnostics["initial_risk"]),
        "random_risk": diagnostics["one_step_random_risk"],
        "oracle_risk": diagnostics["one_step_oracle_risk"],
        "initial_risk": diagnostics["initial_risk"],
    }
