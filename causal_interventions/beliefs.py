"""Finite Bayesian beliefs with full categorical intervention observations."""
from __future__ import annotations

from math import fsum, isfinite
from typing import Protocol, Sequence

from .scoring import excess_log_loss
from .types import Intervention, Query

Belief = tuple[float, ...]


class World(Protocol):
    def intervention_distribution(self, action: Intervention) -> tuple[float, ...]: ...
    def probability(self, query: Query) -> float: ...


def normalize(weights: Sequence[float]) -> Belief:
    values = tuple(float(w) for w in weights)
    if not values or any(not isfinite(w) or w < 0 for w in values):
        raise ValueError("weights must be nonempty, finite, and nonnegative")
    total = fsum(values)
    if not isfinite(total) or total <= 0:
        raise ValueError("weights must have a positive finite total")
    return tuple(w / total for w in values)


class BeliefModel:
    """Public world models only; outcome indices follow each world's table order."""
    def __init__(self, worlds: Sequence[World], actions: Sequence[Intervention],
                 queries: Sequence[Query], prior: Sequence[float] | None = None):
        self.worlds = tuple(worlds)
        self.actions = tuple(sorted(actions))
        self.queries = tuple(queries)
        if not self.worlds or not self.queries:
            raise ValueError("at least one world and query are required")
        variable_orders = [tuple(w.variables) for w in self.worlds if hasattr(w, "variables")]
        if variable_orders and (len(variable_orders) != len(self.worlds)
                                or any(v != variable_orders[0] for v in variable_orders)):
            raise ValueError("worlds must share a canonical variable order for outcomes")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        self.prior = normalize(prior if prior is not None else [1] * len(worlds))
        self._belief(self.prior)
        self.likelihoods = {}
        for action in self.actions:
            rows = tuple(tuple(w.intervention_distribution(action)) for w in worlds)
            if len({len(row) for row in rows}) != 1:
                raise ValueError("world outcome tables must have equal length")
            for row in rows:
                if (not row or any(not isfinite(p) or p < 0 for p in row)
                        or abs(fsum(row) - 1) > 1e-10):
                    raise ValueError("invalid world outcome distribution")
            self.likelihoods[action] = rows
        self.query_probabilities = tuple(tuple(w.probability(q) for q in queries)
                                         for w in worlds)
        if any(not isfinite(p) or not 0 <= p <= 1
               for row in self.query_probabilities for p in row):
            raise ValueError("invalid world query probability")

    def _belief(self, belief: Sequence[float]) -> Belief:
        b = tuple(belief)
        if (len(b) != len(self.worlds) or any(not isfinite(p) or p < 0 for p in b)
                or abs(fsum(b) - 1) > 1e-10):
            raise ValueError("belief must be a normalized probability per world")
        return b

    def outcomes(self, belief: Sequence[float], action: Intervention) -> tuple[float, ...]:
        b = self._belief(belief)
        rows = self.likelihoods[action]
        return tuple(fsum(p * row[o] for p, row in zip(b, rows))
                     for o in range(len(rows[0])))

    def posterior(self, belief: Sequence[float], action: Intervention, outcome: int) -> Belief:
        b = self._belief(belief)
        rows = self.likelihoods[action]
        if isinstance(outcome, bool) or not isinstance(outcome, int) or not 0 <= outcome < len(rows[0]):
            raise ValueError("outcome must be a valid categorical index")
        return normalize(tuple(p * row[outcome] for p, row in zip(b, rows)))

    def predictive(self, belief: Sequence[float]) -> tuple[float, ...]:
        b = self._belief(belief)
        return tuple(fsum(p * row[q] for p, row in zip(b, self.query_probabilities))
                     for q in range(len(self.queries)))

    def risk(self, belief: Sequence[float]) -> float:
        b = self._belief(belief)
        predictions = self.predictive(b)
        return fsum(p * excess_log_loss(row, predictions)
                    for p, row in zip(b, self.query_probabilities) if p > 0)
