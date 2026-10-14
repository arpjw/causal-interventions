"""Development episode loop with deterministic paired outcome draws."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from random import Random
from typing import Protocol

from .analysis import EpisodeResult, EpisodeTruth
from .beliefs import BeliefModel
from .episodes import Episode, canonical_json
from .factorize import assignment_index
from .policies import ExactPolicy, RandomPolicy
from .prompts import HistoryStep, action_id, query_id, render_prompt
from .types import Intervention


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    raw_text: str | None = None


class Backend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int) -> Generation: ...


@dataclass(frozen=True)
class EpisodeRun:
    reference: EpisodeTruth
    results: tuple[EpisodeResult, ...]
    trace: dict[str, object]

    def record(self) -> dict[str, object]:
        return {
            "reference": asdict(self.reference),
            "results": [asdict(result) for result in self.results],
            "trace": self.trace,
        }


def _draw(episode: Episode, action: Intervention, step: int) -> tuple[int, ...]:
    seed = int(sha256(canonical_json([
        episode.outcome_seed, action.target, action.value, step,
    ]).encode()).hexdigest(), 16)
    return episode.worlds[episode.hidden_world].sample_intervention(action, Random(seed))


def _action(raw: str, actions: tuple[Intervention, ...]) -> Intervention | str | None:
    try:
        data = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or set(data) != {"action_id"}:
        return None
    selected = data["action_id"]
    if selected == "STOP":
        return "STOP"
    return next((action for action in actions if action_id(action) == selected), None)


def _forecasts(raw: str, keys: tuple[str, ...]) -> tuple[tuple[float, ...], int]:
    try:
        data = json.loads(raw.strip())
    except (ValueError, TypeError):
        return (0.5,) * len(keys), 1
    if not isinstance(data, dict) or set(data) != {"forecasts"} or not isinstance(data["forecasts"], dict):
        return (0.5,) * len(keys), 1
    values = data["forecasts"]
    invalid = int(set(values) != set(keys))
    result = []
    for key in keys:
        value = values.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or not 0 <= value <= 1:
            invalid = 1
            value = 0.5
        result.append(float(value))
    return tuple(result), invalid


def _baseline(episode: Episode, model: BeliefModel, policy: RandomPolicy | ExactPolicy,
              name: str) -> EpisodeResult:
    belief = model.prior
    for step in range(episode.budget):
        action = policy.choose(belief, episode.budget - step)
        if action is None:
            break
        outcome = _draw(episode, action, step)
        belief = model.posterior(belief, action, assignment_index(outcome))
    truth = tuple(episode.worlds[episode.hidden_world].probability(q) for q in episode.queries)
    return EpisodeResult(episode.episode_id, episode.parameter_id, name, episode.hidden_world,
                         model.predictive(belief), truth)


def run_episode(episode: Episode, backend: Backend, model_id: str,
                action_max_tokens: int = 96, forecast_max_tokens: int = 128,
                allow_locked_test: bool = False, score_actions: bool = False,
                prompt_version: str = "primary-v1") -> EpisodeRun:
    """Run an episode without revealing private world state to the backend.

    Test calls require an explicit flag from a runner that has verified the lock.
    """
    if episode.split != "development" and not (episode.split == "test" and allow_locked_test):
        raise ValueError("test model calls require a verified lock")
    if not model_id or action_max_tokens < 1 or forecast_max_tokens < 1:
        raise ValueError("model ID and token limits must be valid")
    model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
    truth = tuple(episode.worlds[episode.hidden_world].probability(q) for q in episode.queries)
    reference = EpisodeTruth(episode.episode_id, episode.parameter_id, episode.hidden_world, truth)
    history: list[HistoryStep] = []
    calls: list[dict[str, object]] = []
    failures = 0
    remaining = episode.budget
    while remaining:
        prompt = render_prompt(episode, remaining, history, "action", prompt_version)
        generation = backend.generate(prompt, action_max_tokens)
        selected = _action(generation.text, episode.actions)
        calls.append({"phase": "action", "prompt": prompt, **asdict(generation),
                      "parsed_action": action_id(selected) if isinstance(selected, Intervention) else selected})
        if selected == "STOP":
            break
        if selected is None:
            failures += 1
            history.append(HistoryStep(None, None))
        else:
            outcome = _draw(episode, selected, len(history))
            history.append(HistoryStep(selected, outcome))
        remaining -= 1
    prompt = render_prompt(episode, remaining, history, "forecast", prompt_version)
    generation = backend.generate(prompt, forecast_max_tokens)
    submitted, invalid_forecast = _forecasts(generation.text,
                                            tuple(query_id(q) for q in episode.queries))
    failures += invalid_forecast
    calls.append({"phase": "forecast", "prompt": prompt, **asdict(generation),
                  "parsed_forecasts": submitted})
    result = EpisodeResult(episode.episode_id, episode.parameter_id, model_id,
                           episode.hidden_world, submitted, truth, parse_failures=failures)
    random = _baseline(episode, model, RandomPolicy(model, episode.outcome_seed), "random")
    exact = _baseline(episode, model, ExactPolicy(model), "exact")
    results = [result, random, exact]
    if score_actions:
        belief = model.prior
        for step in history:
            if step.action is not None:
                assert step.outcome is not None
                belief = model.posterior(belief, step.action, assignment_index(step.outcome))
        results.append(EpisodeResult(
            episode.episode_id, episode.parameter_id, f"{model_id}:action_bayes",
            episode.hidden_world, model.predictive(belief), truth,
        ))
    return EpisodeRun(reference, tuple(results),
                      {"episode_id": episode.episode_id, "calls": calls,
                       "history": [asdict(step) for step in history],
                       "remaining_budget": remaining, "parse_failures": failures,
                       "input_tokens": sum(call["input_tokens"] for call in calls),
                       "output_tokens": sum(call["output_tokens"] for call in calls),
                       "elapsed_seconds": sum(call["elapsed_seconds"] for call in calls)})
