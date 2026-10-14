"""Exact truncated-factorization enumeration, without sampling error."""

from __future__ import annotations

from .factorize import binary_assignments, factorized_probability
from .types import Intervention


def intervention_distribution(
    variables: tuple[str, ...], order: tuple[str, ...],
    tables: tuple[tuple[float, ...], ...], action: Intervention,
) -> tuple[float, ...]:
    """Return probabilities over full assignments, including the fixed target."""
    if action.target not in variables:
        raise ValueError("intervention target must be a world variable")
    position = variables.index(action.target)
    return tuple(
        factorized_probability(state, variables, order, tables, skip=action.target)
        if state[position] == action.value else 0.0
        for state in binary_assignments(len(variables))
    )
