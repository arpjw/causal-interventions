"""Exact Bernoulli excess log loss, in nats, and reporting-time clipping."""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, log, log1p
from typing import Sequence


def _probability(value: float) -> float:
    value = float(value)
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError("probabilities must be finite and in [0, 1]")
    return value


def bernoulli_kl(truth: float, prediction: float) -> float:
    """KL(Bern(truth) || Bern(prediction)); preserve infinite endpoint loss."""
    p, q = _probability(truth), _probability(prediction)
    if p == q:
        return 0.0
    if q in (0.0, 1.0):
        return float("inf")
    terms = []
    if p > 0:
        terms.append(p * (log(p) - log(q)))
    if p < 1:
        terms.append((1 - p) * (log1p(-p) - log1p(-q)))
    return max(0.0, fsum(terms))


def excess_log_loss(truth: Sequence[float], predictions: Sequence[float]) -> float:
    if not truth or len(truth) != len(predictions):
        raise ValueError("truth and predictions must have equal nonzero length")
    return fsum(bernoulli_kl(p, q) for p, q in zip(truth, predictions)) / len(truth)


@dataclass(frozen=True)
class ForecastScore:
    loss: float
    clipping_count: int
    predictions: tuple[float, ...]


def scored_forecasts(truth: Sequence[float], predictions: Sequence[float],
                     epsilon: float = 1e-6) -> ForecastScore:
    """Clip submitted forecasts only; the exact decision objective is unclipped."""
    if not isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be finite and in (0, 0.5)")
    checked = tuple(_probability(p) for p in predictions)
    clipped = tuple(min(1 - epsilon, max(epsilon, p)) for p in checked)
    return ForecastScore(excess_log_loss(truth, clipped),
                         sum(p != q for p, q in zip(checked, clipped)), clipped)
