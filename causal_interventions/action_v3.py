"""Prospective one-action expected-risk evaluation, independent of sampled outcomes."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from random import Random
from typing import Any, Sequence

from .episodes import Episode, canonical_json, one_step_diagnostics
from .prompts import action_id
from .runner import Backend

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "action-v3.txt"


def action_values(episode: Episode) -> dict[str, float]:
    """Recompute scores from public worlds rather than trusting manifest diagnostics."""
    diagnostics = one_step_diagnostics(episode.worlds, episode.prior,
                                       episode.actions, episode.queries)
    return {action_id(action): risk for action, risk in
            zip(episode.actions, diagnostics["action_risks"], strict=True)}


def render_action_prompt(episode: Episode, assisted: bool = False) -> str:
    if episode.budget != 1:
        raise ValueError("v3 action scoring requires budget one")
    public = episode.public_record()
    choices = list(action_values(episode))
    task: dict[str, Any] = {
        "variables": public["variables"], "joint": public["joint"],
        "candidates": [{"candidate_id": f"W{i}", "order": order}
                       for i, order in enumerate(public["orders"])],
        "prior": public["prior"],
        "allowed_actions": choices, "query": public["queries"][0],
        "response_schema": {"action_id": choices + ["STOP"]},
    }
    if assisted:
        # These are calculated solely from public inputs, and constitute a
        # diagnostic upper bound on value computation, not a hidden label.
        task["computed_action_values"] = {
            "metric": "expected_remaining_bayes_risk_nats_lower_is_better",
            "values": action_values(episode),
        }
    return PROMPT_PATH.read_text(encoding="utf-8").replace(
        "{{PUBLIC_TASK_JSON}}", canonical_json(task))


def parse_action(text: str, episode: Episode) -> tuple[str | None, bool]:
    try:
        value = json.loads(text.strip())
    except (TypeError, ValueError):
        return None, True
    valid = set(action_values(episode)) | {"STOP"}
    if not isinstance(value, dict) or set(value) != {"action_id"}:
        return None, True
    selected = value["action_id"]
    if not isinstance(selected, str) or selected not in valid:
        return None, True
    return selected, False


def run_action(episode: Episode, backend: Backend, assisted: bool,
               max_new_tokens: int = 96, allow_locked_test: bool = False) -> dict[str, Any]:
    if episode.split != "development" and not (episode.split == "test" and allow_locked_test):
        raise ValueError("test model calls require a verified lock")
    if max_new_tokens < 1:
        raise ValueError("token cap must be positive")
    prompt = render_action_prompt(episode, assisted)
    generated = backend.generate(prompt, max_new_tokens)
    selected, invalid = parse_action(generated.text, episode)
    values = action_values(episode)
    diagnostics = one_step_diagnostics(episode.worlds, episode.prior,
                                       episode.actions, episode.queries)
    model_risk = values.get(selected, diagnostics["initial_risk"])
    return {
        "episode_id": episode.episode_id, "parameter_id": episode.parameter_id,
        "split": episode.split, "stratum": episode.stratum,
        "assisted": assisted, "prompt": prompt, "generation": asdict(generated),
        "selected_action": selected, "invalid_action": invalid,
        "model_risk": model_risk,
        "random_risk": diagnostics["one_step_random_risk"],
        "oracle_risk": diagnostics["one_step_oracle_risk"],
        "initial_risk": diagnostics["initial_risk"],
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low = int(index)
    amount = index - low
    return ordered[low] * (1 - amount) + ordered[min(low + 1, len(ordered) - 1)] * amount


def summarize_action_runs(episodes: Sequence[Episode], runs: Sequence[dict[str, Any]],
                          bootstrap_samples: int = 2000, seed: int = 314159) -> dict[str, Any]:
    if not episodes or len(episodes) != len(runs) or bootstrap_samples < 1:
        raise ValueError("complete nonempty rows and a positive bootstrap count required")
    indexed = {row["episode_id"]: row for row in runs}
    if len(indexed) != len(runs) or set(indexed) != {e.episode_id for e in episodes}:
        raise ValueError("runs must match episodes exactly")
    informative = [e for e in episodes if e.stratum.startswith("informative")]
    if not informative:
        raise ValueError("no informative episodes")
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        row = indexed[episode.episode_id]
        if (row["parameter_id"] != episode.parameter_id or row["split"] != episode.split
                or row["stratum"] != episode.stratum):
            raise ValueError("parameter identity mismatch")
        values = action_values(episode)
        selected = row["selected_action"]
        if selected not in values and selected not in ("STOP", None):
            raise ValueError("unrecognized action")
        diagnostics = one_step_diagnostics(episode.worlds, episode.prior,
                                           episode.actions, episode.queries)
        expected = values.get(selected, diagnostics["initial_risk"])
        for key, value in (("model_risk", expected),
                           ("random_risk", diagnostics["one_step_random_risk"]),
                           ("oracle_risk", diagnostics["one_step_oracle_risk"])):
            if abs(row[key] - value) > 1e-12:
                raise ValueError(f"{key} fails independent action-value replay")
        if not episode.stratum.startswith("informative"):
            continue
        groups.setdefault(episode.parameter_id, []).append(row)
    clusters = [{key: fsum(row[key] for row in group) / len(group)
                 for key in ("model_risk", "random_risk", "oracle_risk")}
                for group in groups.values()]
    means = {key: fsum(cluster[key] for cluster in clusters) / len(clusters)
             for key in ("model_risk", "random_risk", "oracle_risk")}
    differences = [cluster["model_risk"] - cluster["random_risk"] for cluster in clusters]
    rng = Random(seed)
    interval = None
    if len(clusters) > 1:
        samples = [fsum(differences[rng.randrange(len(clusters))] for _ in clusters) / len(clusters)
                   for _ in range(bootstrap_samples)]
        interval = [_quantile(samples, 0.025), _quantile(samples, 0.975)]
    opportunity = means["random_risk"] - means["oracle_risk"]
    return {
        "primary_metric": "prior_averaged_one_step_bayes_risk_nats",
        "informative_episodes": len(informative),
        "near_tie_episodes": len(episodes) - len(informative),
        "independent_parameter_clusters": len(clusters), "mean_risks": means,
        "model_minus_random": fsum(differences) / len(differences),
        "interval_95": interval, "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed, "opportunity_gap": opportunity,
        "gap_closure": ((means["random_risk"] - means["model_risk"]) / opportunity
                        if opportunity > 0.01 else None),
        "invalid_actions": sum(bool(row["invalid_action"]) for row in runs),
        "selected_action_counts": {
            **{choice: sum(row["selected_action"] == choice for row in runs)
               for choice in (*action_values(episodes[0]), "STOP")},
            "INVALID": sum(row["selected_action"] is None for row in runs),
        },
        "all_episode_ids_sha256": sha256(canonical_json([e.episode_id for e in episodes]).encode()).hexdigest(),
    }


def compare_action_runs(episodes: Sequence[Episode], first: Sequence[dict[str, Any]],
                        second: Sequence[dict[str, Any]], bootstrap_samples: int = 2000,
                        seed: int = 271828) -> dict[str, Any]:
    """Paired second-minus-first risk on identical informative episodes."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be positive")
    summarize_action_runs(episodes, first, bootstrap_samples=1)
    summarize_action_runs(episodes, second, bootstrap_samples=1)
    a = {row["episode_id"]: row for row in first}
    b = {row["episode_id"]: row for row in second}
    selected = [episode for episode in episodes if episode.stratum.startswith("informative")]
    differences = [b[episode.episode_id]["model_risk"] - a[episode.episode_id]["model_risk"]
                   for episode in selected]
    rng = Random(seed)
    interval = None
    if len(differences) > 1:
        samples = [fsum(differences[rng.randrange(len(differences))] for _ in differences)
                   / len(differences) for _ in range(bootstrap_samples)]
        interval = [_quantile(samples, 0.025), _quantile(samples, 0.975)]
    return {"informative_episodes": len(differences),
            "second_minus_first": fsum(differences) / len(differences),
            "interval_95": interval, "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed}
