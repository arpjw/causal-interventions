"""Frozen primary numeric prompt rendered exclusively from public episode data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Sequence

from .episodes import Episode, canonical_json
from .types import Intervention, Query

TEMPLATE_VERSION = "primary-v1"
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "primary-v1.txt"
TEMPLATE_SHA256 = "31f960493e7cdcca69dcb2e0cba82765a04914c5169e387a38163f2dad80ffba"
V2_TEMPLATE_SHA256 = "467f37ab4e4ca0ca46590e9ee3bf653aea513240bf68457f260134556af9b87c"


@dataclass(frozen=True)
class HistoryStep:
    """An executed action and public result, or one invalid consumed slot."""

    action: Intervention | None
    outcome: tuple[int, ...] | None


def action_id(action: Intervention) -> str:
    return f"do({action.target}={action.value})"


def query_id(query: Query) -> str:
    return f"P({query.outcome}={query.outcome_value}|{action_id(query.intervention)})"


def template_text(version: str = TEMPLATE_VERSION) -> str:
    if version == TEMPLATE_VERSION:
        path, expected = TEMPLATE_PATH, TEMPLATE_SHA256
    elif version == "primary-v2":
        path, expected = TEMPLATE_PATH.with_name("primary-v2.txt"), V2_TEMPLATE_SHA256
    else:
        raise ValueError("unknown prompt version")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("primary prompt template hash mismatch; version changes explicitly")
    return raw.decode("utf-8")


def render_prompt(episode: Episode, remaining_budget: int,
                  history: Sequence[HistoryStep] = (),
                  phase: Literal["action", "forecast"] = "action",
                  version: str = TEMPLATE_VERSION) -> str:
    """Render a deterministic request without episode IDs, private seeds or labels.

    Use forecast phase after STOP or when the budget is exhausted. An invalid
    action is HistoryStep(None, None); raw model text is never copied here.
    """
    if type(remaining_budget) is not int or not 0 <= remaining_budget <= episode.budget:
        raise ValueError("remaining budget must be an integer within the episode budget")
    if len(history) + remaining_budget != episode.budget:
        raise ValueError("history must account for every consumed budget unit")
    if phase not in ("action", "forecast") or (phase == "action" and remaining_budget == 0):
        raise ValueError("invalid phase or action request with exhausted budget")
    public = episode.public_record()
    variables = public["variables"]
    steps = []
    for step in history:
        if step.action is None:
            if step.outcome is not None:
                raise ValueError("invalid action cannot reveal an outcome")
            steps.append({"status": "invalid", "action_id": None, "outcome": None})
            continue
        if step.action not in episode.actions or step.outcome is None:
            raise ValueError("history requires an allowed action and outcome")
        if (len(step.outcome) != len(variables)
                or any(type(v) is not int or v not in (0, 1) for v in step.outcome)
                or step.outcome[variables.index(step.action.target)] != step.action.value):
            raise ValueError("history outcome must be binary and agree with its intervention")
        steps.append({"status": "observed", "action_id": action_id(step.action),
                      "outcome": list(step.outcome)})
    actions = [{"action_id": action_id(a), **item}
               for a, item in zip(episode.actions, public["actions"])]
    queries = [{"query_id": query_id(q), **item}
               for q, item in zip(episode.queries, public["queries"])]
    action_schema = {
        "type": "object", "additionalProperties": False, "required": ["action_id"],
        "properties": {"action_id": {"type": "string", "enum": [a["action_id"] for a in actions] + ["STOP"]}},
    }
    forecast_schema = {
        "type": "object", "additionalProperties": False, "required": ["forecasts"],
        "properties": {"forecasts": {
            "type": "object", "additionalProperties": False,
            "required": [q["query_id"] for q in queries],
            "properties": {q["query_id"]: {"type": "number", "minimum": 0, "maximum": 1} for q in queries},
        }},
    }
    payload = {
        "schema_version": public["schema_version"], "template_version": version,
        "phase": phase, "variables": variables, "joint": public["joint"],
        "candidates": [{"candidate_id": f"W{i}", "order": order} for i, order in enumerate(public["orders"])],
        "prior": public["prior"], "allowed_actions": actions, "queries": queries,
        "initial_budget": public["budget"], "remaining_budget": remaining_budget,
        "history": steps, "response_schema": action_schema if phase == "action" else forecast_schema,
    }
    return template_text(version).replace("{{PUBLIC_TASK_JSON}}", canonical_json(payload))
