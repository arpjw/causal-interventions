"""Frozen analysis contract for paired, independently generated causal worlds.

Invalid or missing forecasts receive 0.5. Every expected episode remains in the
comparison. Joint-parameter clusters receive equal weight, and bootstrap draws
resample clusters rather than repeated outcomes or prompts. Percentile intervals
are descriptive uncertainty summaries; no model performance informed this code.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from typing import Sequence

from .scoring import scored_forecasts

ANALYSIS_CONTRACT = {
    "schema_version": 1,
    "primary_metric": "mean_excess_log_loss_nats",
    "primary_comparison": "model_minus_random",
    "weighting": "equal_joint_parameter_clusters_then_equal_episodes",
    "uncertainty": "paired_cluster_bootstrap_percentile_95",
    "invalid_or_missing_forecast": 0.5,
    "forecast_clipping_epsilon": 1e-6,
    "default_bootstrap_seed": 314159,
    "default_bootstrap_samples": 2000,
    "default_minimum_gap": 1e-6,
    "negative_difference_favors_model": True,
    "claims": "synthetic benchmark only; no causal representation claim",
}


@dataclass(frozen=True)
class EpisodeTruth:
    episode_id: str
    parameter_id: str
    true_world: int
    query_truths: tuple[float, ...]

    def __post_init__(self) -> None:
        if (not self.episode_id or not self.parameter_id or isinstance(self.true_world, bool)
                or not isinstance(self.true_world, int) or self.true_world < 0):
            raise ValueError("invalid episode identity")
        if not self.query_truths or any(not isfinite(p) or not 0 <= p <= 1 for p in self.query_truths):
            raise ValueError("query truths must be nonempty probabilities")


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    parameter_id: str
    policy_id: str
    true_world: int
    submitted_probabilities: tuple[float | None, ...]
    query_truths: tuple[float, ...]
    parse_failures: int = 0
    cost: float = 0.0

    def __post_init__(self) -> None:
        EpisodeTruth(self.episode_id, self.parameter_id, self.true_world, self.query_truths)
        if not self.policy_id or isinstance(self.parse_failures, bool) or not isinstance(self.parse_failures, int) or self.parse_failures < 0:
            raise ValueError("invalid policy ID or parse-failure count")
        if not isfinite(self.cost) or self.cost < 0:
            raise ValueError("cost must be finite and nonnegative")


def _mean(values: Sequence[float]) -> float:
    return fsum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[min(low + 1, len(ordered) - 1)] * fraction


def analyze_paired(reference: Sequence[EpisodeTruth], results: Sequence[EpisodeResult],
                   model_id: str, random_id: str = "random", oracle_id: str = "exact",
                   seed: int = 314159, bootstrap_samples: int = 2000,
                   minimum_gap: float = 1e-6) -> dict[str, object]:
    """Analyze three paired policies, retaining every reference episode.

    A missing model row is a recorded failure with one 0.5 forecast per query.
    Missing or malformed baseline rows raise a protocol error. Invalid model
    entries and absent entries within a model row become 0.5. Extra query
    entries are flagged as malformed but cannot shift the registered query set.
    Identity or ground-truth corruption raises instead of changing the estimand.
    """
    policies = (model_id, random_id, oracle_id)
    if len(set(policies)) != 3 or any(not p for p in policies):
        raise ValueError("three distinct nonempty policy IDs required")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    if not isfinite(minimum_gap) or minimum_gap < 0:
        raise ValueError("minimum gap must be finite and nonnegative")
    expected = {e.episode_id: e for e in reference}
    if not expected or len(expected) != len(reference):
        raise ValueError("reference must contain unique episodes")
    indexed = {}
    for row in results:
        if row.episode_id not in expected or row.policy_id not in policies:
            raise ValueError("unexpected episode or policy result")
        truth = expected[row.episode_id]
        if (row.parameter_id, row.true_world, row.query_truths) != (truth.parameter_id, truth.true_world, truth.query_truths):
            raise ValueError("result identity or query truths disagree with reference")
        key = row.episode_id, row.policy_id
        if key in indexed:
            raise ValueError("duplicate episode-policy result")
        indexed[key] = row
    totals = {p: {"missing_rows": 0, "invalid_forecasts": 0, "malformed_rows": 0,
                  "parse_failures": 0, "clipping_count": 0, "cost": 0.0} for p in policies}
    cluster_losses: dict[str, dict[str, list[float]]] = {}
    episode_scores = []
    for eid in sorted(expected):
        truth = expected[eid]
        group = cluster_losses.setdefault(truth.parameter_id, {p: [] for p in policies})
        for policy in policies:
            row = indexed.get((eid, policy))
            submitted = row.submitted_probabilities if row else ()
            missing = int(row is None)
            invalid = 0
            forecasts = []
            for i in range(len(truth.query_truths)):
                value = submitted[i] if i < len(submitted) else None
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
                    invalid += 1
                    value = 0.5
                forecasts.append(float(value))
            score = scored_forecasts(truth.query_truths, forecasts)
            malformed = int(row is not None and (invalid > 0 or len(submitted) != len(truth.query_truths)))
            if policy != model_id and (missing or malformed or row.parse_failures):
                raise ValueError("protocol failure: baseline rows must be complete and valid")
            counts = totals[policy]
            counts["missing_rows"] += missing
            counts["invalid_forecasts"] += invalid
            counts["malformed_rows"] += malformed
            counts["parse_failures"] += row.parse_failures if row else 0
            counts["clipping_count"] += score.clipping_count
            counts["cost"] += row.cost if row else 0
            group[policy].append(score.loss)
            episode_scores.append({"episode_id": eid, "parameter_id": truth.parameter_id,
                                   "policy_id": policy, "loss": score.loss,
                                   "missing_row": bool(missing), "invalid_forecasts": invalid})
    clusters = [{p: _mean(group[p]) for p in policies}
                for _, group in sorted(cluster_losses.items())]
    means = {p: _mean([c[p] for c in clusters]) for p in policies}
    differences = [c[model_id] - c[random_id] for c in clusters]
    rng = Random(seed)
    # A single independent draw cannot estimate population uncertainty.
    interval = None
    if len(clusters) >= 2:
        samples = [_mean([differences[rng.randrange(len(clusters))] for _ in clusters])
                   for _ in range(bootstrap_samples)]
        interval = [_quantile(samples, 0.025), _quantile(samples, 0.975)]
    gap = means[random_id] - means[oracle_id]
    baseline_complete = all(totals[p]["malformed_rows"] == 0 and totals[p]["parse_failures"] == 0
                            for p in (random_id, oracle_id))
    closure = ((means[random_id] - means[model_id]) / gap
               if baseline_complete and gap > minimum_gap else None)
    return {"contract": ANALYSIS_CONTRACT.copy(), "episodes": len(reference),
            "independent_parameter_clusters": len(clusters), "mean_losses": means,
            "paired_model_minus_random": _mean(differences),
            "paired_difference_interval_95": interval,
            "bootstrap_seed": seed, "bootstrap_samples": bootstrap_samples,
            "minimum_gap": minimum_gap, "opportunity_gap": gap, "gap_closure": closure,
            "baseline_complete": baseline_complete, "counts": totals,
            "episode_scores": episode_scores}
