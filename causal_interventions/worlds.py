"""Two causal factorizations of the same observational distribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum
from random import Random
from typing import Literal

from .factorize import binary_assignments, factorize_joint, factorized_probability
from .interventions import intervention_distribution
from .types import Intervention, Query


@dataclass(frozen=True)
class FiniteSCM:
    """Binary SCM whose complete DAG follows ``order``.

    CPTs come from the shared observational ``joint``; interventions omit the
    target's CPT. Assignment indices always follow ``variables``, not ``order``.
    """

    variables: tuple[str, ...]
    order: tuple[str, ...]
    joint: tuple[float, ...]
    tables: tuple[tuple[float, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "order", tuple(self.order))
        object.__setattr__(self, "joint", tuple(self.joint))
        object.__setattr__(self, "tables", factorize_joint(self.variables, self.order, self.joint))

    def observational_probability(self, assignment: tuple[int, ...]) -> float:
        return factorized_probability(assignment, self.variables, self.order, self.tables)

    def intervention_distribution(self, action: Intervention) -> tuple[float, ...]:
        return intervention_distribution(self.variables, self.order, self.tables, action)

    def probability(self, query: Query) -> float:
        if query.outcome not in self.variables:
            raise ValueError("query outcome must be a world variable")
        position = self.variables.index(query.outcome)
        return fsum(
            p for state, p in zip(binary_assignments(len(self.variables)),
                                  self.intervention_distribution(query.intervention))
            if state[position] == query.outcome_value
        )

    def sample_intervention(self, action: Intervention, rng: Random) -> tuple[int, ...]:
        """Sample one complete assignment from the exact intervention law."""
        distribution = self.intervention_distribution(action)
        draw = rng.random()
        cumulative = 0.0
        last_positive = 0
        for index, p in enumerate(distribution):
            if p > 0:
                last_positive = index
            cumulative += p
            if draw < cumulative:
                return binary_assignments(len(self.variables))[index]
        return binary_assignments(len(self.variables))[last_positive]

Variable = Literal["X", "Y"]
Joint = tuple[float, float, float, float]  # P(00), P(01), P(10), P(11)


@dataclass(frozen=True)
class BinaryWorld:
    joint: Joint
    cause: Variable

    def __post_init__(self) -> None:
        if self.cause not in ("X", "Y"):
            raise ValueError("cause must be X or Y")
        if any(p <= 0 for p in self.joint) or abs(sum(self.joint) - 1.0) > 1e-10:
            raise ValueError("joint must have positive cells that sum to one")

    def observational_probability(self, x: int, y: int) -> float:
        if x not in (0, 1) or y not in (0, 1):
            raise ValueError("X and Y must be binary")
        return self.joint[2 * x + y]

    def intervention_probability(self, target: Variable, value: int) -> float:
        """Return P(Y=1 | do(target=value))."""
        if target not in ("X", "Y") or value not in (0, 1):
            raise ValueError("target and value must be a valid binary intervention")
        if target == "Y":
            return float(value)
        if self.cause == "X":
            row = self.joint[2 * value : 2 * value + 2]
            return row[1] / sum(row)
        return self.joint[1] + self.joint[3]

    def sample_observation(self, rng: Random) -> tuple[int, int]:
        draw = rng.random()
        cumulative = 0.0
        for index, probability in enumerate(self.joint):
            cumulative += probability
            if draw < cumulative:
                return divmod(index, 2)
        return (1, 1)

    def sample_intervention(self, target: Variable, value: int, rng: Random) -> int:
        """Sample the resulting Y value after an intervention."""
        return int(rng.random() < self.intervention_probability(target, value))


DEMO_JOINT: Joint = (0.4, 0.1, 0.1, 0.4)


def demo_worlds() -> tuple[BinaryWorld, BinaryWorld]:
    """Return observationally identical X→Y and Y→X worlds."""
    return BinaryWorld(DEMO_JOINT, "X"), BinaryWorld(DEMO_JOINT, "Y")
