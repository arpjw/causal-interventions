"""Deterministic, model-blind three-variable benchmark construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import combinations, permutations, product
import json
import math
from random import Random
from typing import Any

from .types import Intervention, Query
from .worlds import FiniteSCM

VARIABLES = ("X", "Y", "Z")
STATES = tuple(product((0, 1), repeat=3))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class Stratum:
    name: str
    candidate_count: int = 3
    budget: int = 1
    min_query_range: float = 0.02
    max_query_range: float = 1.0
    min_action_gap: float = 0.00001
    max_action_gap: float = 1.0
    min_second_action_gain: float = 0.000001
    max_initial_risk: float = 1.0
    min_opportunity_gap: float = 0.001

    def __post_init__(self) -> None:
        if not self.name or self.candidate_count not in range(2, 7) or self.budget not in (1, 2, 3):
            raise ValueError("invalid stratum name, candidate count, or budget")
        bounds = (self.min_query_range, self.max_query_range, self.min_action_gap,
                  self.max_action_gap, self.min_second_action_gain, self.max_initial_risk, self.min_opportunity_gap)
        if any(not math.isfinite(v) or v < 0 for v in bounds):
            raise ValueError("stratum thresholds must be finite and nonnegative")
        if self.min_query_range > self.max_query_range or self.min_action_gap > self.max_action_gap:
            raise ValueError("stratum bounds are reversed")


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 271828
    analysis_version: int = 1
    coefficient_bound: float = 0.6
    min_cell: float = 0.002
    max_attempts_per_stratum: int = 10000
    strata: tuple[Stratum, ...] = (Stratum("informative"), Stratum(
        "near_tie", min_action_gap=0.0, max_action_gap=0.00001,
        min_second_action_gain=0.0, min_opportunity_gap=0.0))

    def __post_init__(self) -> None:
        if self.analysis_version not in (1, 2, 3, 4):
            raise ValueError("unsupported analysis version")
        if not math.isfinite(self.coefficient_bound) or self.coefficient_bound <= 0:
            raise ValueError("coefficient bound must be finite and positive")
        if not 0 < self.min_cell < 0.125 or self.max_attempts_per_stratum < 1:
            raise ValueError("invalid cell bound or attempt limit")
        if not self.strata or len({s.name for s in self.strata}) != len(self.strata):
            raise ValueError("strata must have distinct names")


@dataclass(frozen=True)
class Episode:
    episode_id: str
    parameter_id: str
    split: str
    stratum: str
    generator_seed: int
    parameters: tuple[float, ...]
    worlds: tuple[FiniteSCM, ...]
    prior: tuple[float, ...]
    actions: tuple[Intervention, ...]
    queries: tuple[Query, ...]
    budget: int
    hidden_world: int
    outcome_seed: int
    diagnostics: dict[str, Any]

    def public_record(self) -> dict[str, Any]:
        """Only information available to a policy; hidden world and RNG stay private."""
        return {
            "schema_version": 1,
            "variables": list(VARIABLES), "joint": list(self.worlds[0].joint),
            "orders": [list(w.order) for w in self.worlds], "prior": list(self.prior),
            "actions": [asdict(a) for a in self.actions],
            "queries": [asdict(q) for q in self.queries], "budget": self.budget,
        }

    def record(self) -> dict[str, Any]:
        result = self.public_record()
        result.update(episode_id=self.episode_id, parameter_id=self.parameter_id, split=self.split, stratum=self.stratum,
                      generator_seed=self.generator_seed, parameters=list(self.parameters),
                      hidden_world=self.hidden_world, outcome_seed=self.outcome_seed,
                      diagnostics=self.diagnostics)
        result["validation_hash"] = digest(result)
        return result


def joint_from_parameters(parameters: tuple[float, ...]) -> tuple[float, ...]:
    """Seven spin interactions, in subset-size then lexicographic order."""
    if len(parameters) != 7 or any(not math.isfinite(v) for v in parameters):
        raise ValueError("seven finite log-linear parameters required")
    subsets = tuple(s for size in (1, 2, 3) for s in combinations(range(3), size))
    logits = [sum(c * math.prod(2 * state[i] - 1 for i in subset)
                  for c, subset in zip(parameters, subsets)) for state in STATES]
    weights = [math.exp(v - max(logits)) for v in logits]
    return tuple(v / sum(weights) for v in weights)


def _entropy(p: float) -> float:
    return -sum(v * math.log(v) for v in (p, 1 - p) if v > 0)


def one_step_diagnostics(worlds: tuple[FiniteSCM, ...], prior: tuple[float, ...],
                         actions: tuple[Intervention, ...], queries: tuple[Query, ...]) -> dict[str, Any]:
    """Independent entropy-based exact risk calculation used for generator gates."""
    probabilities = tuple(tuple(w.probability(q) for q in queries) for w in worlds)

    def risk(belief: tuple[float, ...]) -> float:
        return max(0.0, sum(_entropy(sum(b * p[q] for b, p in zip(belief, probabilities)))
                           - sum(b * _entropy(p[q]) for b, p in zip(belief, probabilities))
                           for q in range(len(queries))) / len(queries))

    initial = risk(prior)
    risks = []
    for action in actions:
        likelihoods = [w.intervention_distribution(action) for w in worlds]
        expected = 0.0
        for outcome in range(8):
            mass = sum(b * likelihoods[w][outcome] for w, b in enumerate(prior))
            if mass > 0:
                posterior = tuple(b * likelihoods[w][outcome] / mass for w, b in enumerate(prior))
                expected += mass * risk(posterior)
        risks.append(expected)
    gains = sorted((initial - r for r in risks), reverse=True)
    return {"initial_risk": initial, "action_risks": risks,
            "action_gap": max(risks) - min(risks), "second_action_gain": gains[1],
            "one_step_random_risk": sum(risks) / len(risks),
            "one_step_oracle_risk": min([initial] + risks),
            "one_step_opportunity_gap": sum(risks) / len(risks) - min([initial] + risks),
            "query_range": max(max(p[q] for p in probabilities) - min(p[q] for p in probabilities)
                               for q in range(len(queries)))}


def rejection_reason(joint: tuple[float, ...], diagnostics: dict[str, Any],
                     config: GeneratorConfig, stratum: Stratum) -> str | None:
    if min(joint) < config.min_cell:
        return "minimum_cell"
    for metric, lower, upper in (("query_range", stratum.min_query_range, stratum.max_query_range),
                                  ("action_gap", stratum.min_action_gap, stratum.max_action_gap)):
        if not lower <= diagnostics[metric] <= upper:
            return metric
    if diagnostics["second_action_gain"] < stratum.min_second_action_gain:
        return "second_action_gain"
    if diagnostics["one_step_opportunity_gap"] < stratum.min_opportunity_gap:
        return "opportunity_gap"
    if diagnostics["initial_risk"] > stratum.max_initial_risk:
        return "initial_risk"
    return None


def generate_split(config: GeneratorConfig, split: str, per_stratum: int) -> tuple[tuple[Episode, ...], dict[str, Any]]:
    if split not in ("development", "test") or per_stratum < 1:
        raise ValueError("split must be development or test and count must be positive")
    episodes = []
    counts: dict[str, Any] = {}
    used_parameters: set[str] = set()
    for stratum in config.strata:
        stream_seed = int(digest([config.seed, split, stratum.name]), 16)
        rng = Random(stream_seed)
        rejected: dict[str, int] = {}
        accepted = 0
        optimal_action_counts = [0, 0, 0]
        for attempt in range(config.max_attempts_per_stratum):
            parameters = tuple(rng.uniform(-config.coefficient_bound, config.coefficient_bound) for _ in range(7))
            joint = joint_from_parameters(parameters)
            orders = rng.sample(list(permutations(VARIABLES)), stratum.candidate_count)
            worlds = tuple(FiniteSCM(VARIABLES, order, joint) for order in orders)
            prior = tuple(1 / len(worlds) for _ in worlds)
            queries = (Query(Intervention("X", 1), "Z"),)
            actions = (Intervention("X", 0), Intervention("Y", 0), Intervention("Y", 1))
            diagnostics = one_step_diagnostics(worlds, prior, actions, queries)
            parameter_id = digest(parameters)
            reason = rejection_reason(joint, diagnostics, config, stratum)
            if (reason is None and config.analysis_version in (3, 4)
                    and stratum.name.startswith("informative")):
                best = min(range(len(actions)), key=lambda index: diagnostics["action_risks"][index])
                if optimal_action_counts[best] >= math.ceil(per_stratum / len(actions)):
                    reason = "optimal_action_quota"
            if parameter_id in used_parameters:
                reason = "duplicate_parameter"
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            used_parameters.add(parameter_id)
            if config.analysis_version in (3, 4) and stratum.name.startswith("informative"):
                optimal_action_counts[best] += 1
            episode_id = digest([split, stratum.name, parameter_id, orders, stratum.budget])
            episodes.append(Episode(episode_id, parameter_id, split, stratum.name, stream_seed,
                                    parameters, worlds, prior, actions, queries, stratum.budget,
                                    rng.randrange(len(worlds)), rng.getrandbits(64), diagnostics))
            accepted += 1
            if accepted == per_stratum:
                break
        counts[stratum.name] = {"accepted": accepted, "rejected": rejected,
                                "attempted": accepted + sum(rejected.values())}
        if accepted != per_stratum:
            raise ValueError(f"stratum {stratum.name} exhausted attempt limit: {counts[stratum.name]}")
    return tuple(episodes), counts
