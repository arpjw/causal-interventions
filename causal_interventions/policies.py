"""Exact local experiment design; STOP is represented by None.

Ties within 1e-14 nats prefer STOP, then lexicographic intervention order.
Caches retain full precision beliefs rather than merging rounded states.
"""
from __future__ import annotations

from math import fsum, log
from random import Random
from typing import Sequence

from .beliefs import Belief, BeliefModel
from .types import Intervention

TIE_TOLERANCE = 1e-14


def _budget(budget: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")


def entropy(belief: Sequence[float]) -> float:
    return -fsum(p * log(p) for p in belief if p > 0)


def one_step_risk(model: BeliefModel, belief: Belief, action: Intervention) -> float:
    return fsum(p * model.risk(model.posterior(belief, action, o))
                for o, p in enumerate(model.outcomes(belief, action)) if p > 0)


class RandomPolicy:
    def __init__(self, model: BeliefModel, seed: int = 0):
        self.model = model
        self.rng = Random(seed)

    def choose(self, belief: Belief, budget: int) -> Intervention | None:
        _budget(budget)
        self.model._belief(belief)
        return self.rng.choice(self.model.actions) if budget and self.model.actions else None


class GreedyInformationGainPolicy:
    def __init__(self, model: BeliefModel):
        self.model = model

    def scores(self, belief: Belief) -> dict[Intervention, float]:
        b = self.model._belief(belief)
        return {a: entropy(b) - fsum(p * entropy(self.model.posterior(b, a, o))
                                    for o, p in enumerate(self.model.outcomes(b, a)) if p > 0)
                for a in self.model.actions}

    def choose(self, belief: Belief, budget: int) -> Intervention | None:
        _budget(budget)
        self.model._belief(belief)
        best, gain = None, 0.0
        if budget:
            for action, score in self.scores(belief).items():
                if score > gain + TIE_TOLERANCE:
                    best, gain = action, score
        return best


class GreedyQueryRiskPolicy:
    def __init__(self, model: BeliefModel):
        self.model = model

    def scores(self, belief: Belief) -> dict[Intervention, float]:
        risk = self.model.risk(belief)
        return {a: risk - one_step_risk(self.model, belief, a) for a in self.model.actions}

    def choose(self, belief: Belief, budget: int) -> Intervention | None:
        _budget(budget)
        self.model._belief(belief)
        best, gain = None, 0.0
        if budget:
            for action, score in self.scores(belief).items():
                if score > gain + TIE_TOLERANCE:
                    best, gain = action, score
        return best


class ExactPolicy:
    def __init__(self, model: BeliefModel):
        self.model = model
        self._cache: dict[tuple[Belief, int], tuple[float, Intervention | None]] = {}

    def action_values(self, belief: Belief, budget: int) -> dict[Intervention | None, float]:
        _budget(budget)
        b = self.model._belief(belief)
        values: dict[Intervention | None, float] = {None: self.model.risk(b)}
        if budget:
            for a in self.model.actions:
                values[a] = fsum(p * self.value(self.model.posterior(b, a, o), budget - 1)
                                 for o, p in enumerate(self.model.outcomes(b, a)) if p > 0)
        return values

    def _solve(self, belief: Belief, budget: int) -> tuple[float, Intervention | None]:
        _budget(budget)
        b = self.model._belief(belief)
        key = (b, budget)
        if key not in self._cache:
            values = self.action_values(b, budget)
            minimum = min(values.values())
            action = next(a for a, value in values.items() if value <= minimum + TIE_TOLERANCE)
            # Return the actual chosen policy's risk, including deterministic ties.
            self._cache[key] = (values[action], action)
        return self._cache[key]

    def value(self, belief: Belief, budget: int) -> float:
        return self._solve(belief, budget)[0]

    def choose(self, belief: Belief, budget: int) -> Intervention | None:
        return self._solve(belief, budget)[1]


Policy = RandomPolicy | GreedyInformationGainPolicy | GreedyQueryRiskPolicy | ExactPolicy


def expected_policy_risk(model: BeliefModel, policy: Policy, belief: Belief, budget: int) -> float:
    """Integrate all outcome branches and random choices without advancing RNG."""
    _budget(budget)
    b = model._belief(belief)
    if policy.model is not model:
        raise ValueError("policy and evaluation model must be identical")
    if not budget or not model.actions:
        return model.risk(b)
    if isinstance(policy, ExactPolicy):
        return policy.value(b, budget)
    actions = model.actions if isinstance(policy, RandomPolicy) else (policy.choose(b, budget),)
    return fsum(model.risk(b) if a is None else
                fsum(p * expected_policy_risk(model, policy, model.posterior(b, a, o), budget - 1)
                     for o, p in enumerate(model.outcomes(b, a)) if p > 0)
                for a in actions) / len(actions)
