"""Shared, immutable identifiers for interventions and evaluation queries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Intervention:
    target: str
    value: int

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("intervention target must be nonempty")
        if type(self.value) is not int or self.value not in (0, 1):
            raise ValueError("intervention value must be binary")


@dataclass(frozen=True, order=True)
class Query:
    intervention: Intervention
    outcome: str
    outcome_value: int = 1

    def __post_init__(self) -> None:
        if not self.outcome:
            raise ValueError("query outcome must be nonempty")
        if type(self.outcome_value) is not int or self.outcome_value not in (0, 1):
            raise ValueError("query outcome value must be binary")
