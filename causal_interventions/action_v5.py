"""Stage-level public-input causal calculation and calculator-agent contracts."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import fsum, isfinite, log
from pathlib import Path
from random import Random
from typing import Any

from .action_v3 import summarize_action_runs
from .beliefs import BeliefModel
from .episodes import Episode, STATES, canonical_json
from .policies import one_step_risk
from .prompts import action_id
from .runner import Backend, Generation

ROOT = Path(__file__).resolve().parents[1]
STAGE_PROMPT = ROOT / "prompts/action-v5-stages.txt"
TOOL_PROMPT = ROOT / "prompts/action-v5-tool.txt"
CONDITIONS = ("q32_raw_stages", "q32_calculator_agent", "q235_short_raw_stages",
              "q235_long_raw_stages", "q235_likelihood_stages")
STAGE_CONDITIONS = tuple(name for name in CONDITIONS if name != "q32_calculator_agent")


def action_labels(episode: Episode) -> dict[str, str]:
    if len(episode.actions) != 3:
        raise ValueError("v5 requires three actions")
    semantics = [action_id(action) for action in episode.actions]
    seed = int(sha256((episode.episode_id + ":v5-opaque").encode()).hexdigest(), 16)
    Random(seed).shuffle(semantics)
    return dict(zip("ABC", semantics, strict=True))


def _action_by_label(episode: Episode, label: str):
    semantic = action_labels(episode)[label]
    return next(action for action in episode.actions if action_id(action) == semantic)


def probe_index(episode: Episode) -> int:
    action = _action_by_label(episode, "A")
    target = list(episode.worlds[0].variables).index(action.target)
    return next(index for index, state in enumerate(STATES) if state[target] == action.value)


def stage_truth(episode: Episode) -> dict[str, Any]:
    model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
    labels = action_labels(episode)
    probe = probe_index(episode)
    action_a = _action_by_label(episode, "A")
    posterior = model.posterior(model.prior, action_a, probe)
    return {
        "query_probabilities": {f"W{i}": model.query_probabilities[i][0]
                                for i in range(len(model.worlds))},
        "probe_likelihoods": {f"W{i}": model.likelihoods[action_a][i][probe]
                              for i in range(len(model.worlds))},
        "posterior_probabilities": {f"W{i}": posterior[i]
                                    for i in range(len(model.worlds))},
        "action_risks": {label: one_step_risk(model, model.prior,
                                               _action_by_label(episode, label))
                         for label in labels},
        "random_risk": fsum(one_step_risk(model, model.prior, action)
                             for action in episode.actions) / len(episode.actions),
        "oracle_risk": min(model.risk(model.prior), *(one_step_risk(model, model.prior, action)
                                                       for action in episode.actions)),
        "initial_risk": model.risk(model.prior),
    }


def public_task(episode: Episode, *, likelihoods: bool) -> dict[str, Any]:
    if episode.budget != 1 or len(episode.queries) != 1:
        raise ValueError("v5 requires one action and one query")
    public = episode.public_record()
    labels = action_labels(episode)
    task: dict[str, Any] = {
        "variables": public["variables"], "joint": public["joint"],
        "candidates": [{"candidate_id": f"W{i}", "order": order}
                       for i, order in enumerate(public["orders"])],
        "prior": public["prior"],
        "allowed_actions": [{"action_id": label, "intervention": semantic}
                            for label, semantic in labels.items()],
        "query": public["queries"][0],
        "probe": {"action_id": "A", "outcome_state_lexicographic": list(STATES[probe_index(episode)])},
    }
    if likelihoods:
        truth = stage_truth(episode)
        task["intermediate_distributions"] = {
            "outcome_states_lexicographic": [list(state) for state in STATES],
            "query_probability_by_candidate": truth["query_probabilities"],
            "experiment_outcome_probability_by_action_and_candidate": {
                label: {f"W{i}": list(world.intervention_distribution(
                    _action_by_label(episode, label)))
                        for i, world in enumerate(episode.worlds)}
                for label in labels},
        }
    return task


def _number_object(keys: list[str]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": keys, "properties": {key: {"type": "number"} for key in keys}}


def stage_schema(episode: Episode, condition: str) -> dict[str, Any]:
    if condition not in STAGE_CONDITIONS:
        raise ValueError("stage condition required")
    worlds = [f"W{i}" for i in range(len(episode.worlds))]
    properties = {
        "posterior_probabilities": _number_object(worlds),
        "action_risks": _number_object(list("ABC")),
        "action_id": {"type": "string", "enum": [*"ABC", "STOP"]},
    }
    if condition != "q235_likelihood_stages":
        properties["query_probabilities"] = _number_object(worlds)
        properties["probe_likelihoods"] = _number_object(worlds)
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def render_stage_prompt(episode: Episode, condition: str) -> str:
    schema = stage_schema(episode, condition)
    task = public_task(episode, likelihoods=condition == "q235_likelihood_stages")
    task["response_schema"] = schema
    return STAGE_PROMPT.read_text(encoding="utf-8").replace(
        "{{PUBLIC_TASK_JSON}}", canonical_json(task))


def render_tool_prompt(episode: Episode, transcript: list[dict[str, Any]]) -> str:
    task = public_task(episode, likelihoods=False)
    task.pop("probe")
    return (TOOL_PROMPT.read_text(encoding="utf-8")
            .replace("{{PUBLIC_TASK_JSON}}", canonical_json(task))
            .replace("{{REMAINING_TOOL_CALLS}}", str(max(0, 3 - len(transcript))))
            .replace("{{TOOL_TRANSCRIPT_JSON}}", canonical_json(transcript)))


def tool_schema(*, final_only: bool = False) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False,
            "required": ["operation", "action_id"],
            "properties": {
                "operation": {"type": "string", "enum": ["final"] if final_only else ["evaluate", "final"]},
                "action_id": {"type": "string", "enum": [*"ABC", "STOP"]},
            }}


def _numeric_stage(value: Any, keys: list[str], maximum: float) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(keys):
        return None
    if any(isinstance(value[key], bool) or not isinstance(value[key], (int, float))
           or not isfinite(value[key]) or not 0 <= value[key] <= maximum for key in keys):
        return None
    return {key: float(value[key]) for key in keys}


def score_stage(episode: Episode, condition: str, generation: Generation,
                *, allow_locked_test: bool = False) -> dict[str, Any]:
    if episode.split != "development" and not (episode.split == "test" and allow_locked_test):
        raise ValueError("test scoring requires verified lock")
    schema = stage_schema(episode, condition)
    prompt = render_stage_prompt(episode, condition)
    truth = stage_truth(episode)
    try:
        response = json.loads(generation.text.strip())
    except (TypeError, ValueError):
        response = None
    strict = isinstance(response, dict) and set(response) == set(schema["required"])
    observed: dict[str, dict[str, float] | None] = {}
    errors: dict[str, float | None] = {}
    for name in ("query_probabilities", "probe_likelihoods", "posterior_probabilities",
                 "action_risks"):
        if name not in schema["required"]:
            continue
        keys = list(truth[name])
        observed[name] = _numeric_stage(response.get(name) if strict else None, keys,
                                        log(2) + 1e-12 if name == "action_risks" else 1)
        errors[name] = (fsum(abs(observed[name][key] - truth[name][key]) for key in keys) / len(keys)
                        if observed[name] is not None else None)
    label = response.get("action_id") if strict else None
    if label not in (*"ABC", "STOP"):
        label = None
    labels = action_labels(episode)
    selected = labels.get(label, "STOP") if label is not None else None
    risks = observed["action_risks"]
    self_consistent = (isinstance(label, str) and label in "ABC" and risks is not None
                       and risks[label] <= min(risks.values()) + 1e-10)
    return {
        "episode_id": episode.episode_id, "parameter_id": episode.parameter_id,
        "split": episode.split, "stratum": episode.stratum, "condition": condition,
        "prompt": prompt, "generation": asdict(generation),
        "selected_label": label, "selected_action": selected,
        "invalid_action": selected is None,
        "stage_values": observed, "stage_mae": errors,
        "self_consistent": bool(self_consistent),
        "model_risk": truth["action_risks"].get(label, truth["initial_risk"]),
        "random_risk": truth["random_risk"], "oracle_risk": truth["oracle_risk"],
        "initial_risk": truth["initial_risk"],
    }


def run_tool_agent(episode: Episode, backend: Backend, *, allow_locked_test: bool = False) -> dict[str, Any]:
    if episode.split != "development" and not (episode.split == "test" and allow_locked_test):
        raise ValueError("test calls require verified lock")
    truth = stage_truth(episode)
    transcript: list[dict[str, Any]] = []
    calls = []
    final: str | None = None
    for step in range(4):
        prompt = render_tool_prompt(episode, transcript)
        generation = backend.generate(prompt, 96)
        try:
            response = json.loads(generation.text.strip())
        except (TypeError, ValueError):
            response = None
        calls.append({"prompt": prompt, "generation": asdict(generation)})
        if (not isinstance(response, dict) or set(response) != {"operation", "action_id"}
                or response["action_id"] not in (*"ABC", "STOP")
                or response["operation"] not in ("evaluate", "final")):
            break
        if response["operation"] == "final":
            final = response["action_id"]
            break
        if step == 3 or response["action_id"] == "STOP":
            break
        label = response["action_id"]
        transcript.append({"operation": "evaluate_action", "action_id": label,
                           "expected_risk_nats": truth["action_risks"][label]})
    labels = action_labels(episode)
    selected = labels.get(final, "STOP") if final is not None else None
    return {
        "episode_id": episode.episode_id, "parameter_id": episode.parameter_id,
        "split": episode.split, "stratum": episode.stratum,
        "condition": "q32_calculator_agent", "calls": calls,
        "tool_transcript": transcript, "selected_label": final,
        "selected_action": selected, "invalid_action": selected is None,
        "distinct_tool_actions": len({call["action_id"] for call in transcript}),
        "model_risk": truth["action_risks"].get(final, truth["initial_risk"]),
        "random_risk": truth["random_risk"], "oracle_risk": truth["oracle_risk"],
        "initial_risk": truth["initial_risk"],
    }


def summarize(episodes: list[Episode], rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize_action_runs(episodes, rows)
    if rows[0]["condition"] == "q32_calculator_agent":
        result["mean_distinct_tool_actions"] = fsum(row["distinct_tool_actions"] for row in rows) / len(rows)
        result["all_three_tool_actions"] = sum(row["distinct_tool_actions"] == 3 for row in rows)
    else:
        names = sorted({name for row in rows for name in row["stage_mae"]})
        result["stage_mae"] = {
            name: {
                "valid": sum(row["stage_mae"][name] is not None for row in rows),
                "mean_absolute_error": (fsum(row["stage_mae"][name] for row in rows
                                            if row["stage_mae"][name] is not None)
                                        / sum(row["stage_mae"][name] is not None for row in rows)
                                        if any(row["stage_mae"][name] is not None for row in rows) else None),
            } for name in names}
        result["self_consistent_actions"] = sum(row["self_consistent"] for row in rows)
    return result
