"""Finite binary chain-rule factorizations in an explicit variable order.

Joint cells use lexicographic assignments in ``variables`` order (the first
variable is the most significant bit). Each CPT stores P(node=1 | prefix),
with parent prefixes indexed in the world's causal order.
"""

from __future__ import annotations

from itertools import product
from math import fsum, isfinite


def binary_assignments(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(product((0, 1), repeat=size))


def assignment_index(assignment: tuple[int, ...]) -> int:
    index = 0
    for bit in assignment:
        if not isinstance(bit, int) or bit not in (0, 1):
            raise ValueError("assignment values must be binary")
        index = 2 * index + bit
    return index


def validate_joint(
    variables: tuple[str, ...], order: tuple[str, ...], joint: tuple[float, ...]
) -> None:
    if not variables or any(not isinstance(v, str) or not v for v in variables):
        raise ValueError("variables must be nonempty names")
    if len(set(variables)) != len(variables):
        raise ValueError("variable names must be unique")
    if len(order) != len(variables) or set(order) != set(variables):
        raise ValueError("order must be a permutation of variables")
    if len(joint) != 2 ** len(variables):
        raise ValueError("joint must contain one cell per binary assignment")
    if any(not isfinite(p) or p <= 0 for p in joint):
        raise ValueError("joint cells must be finite and strictly positive")
    if abs(fsum(joint) - 1.0) > 1e-10:
        raise ValueError("joint cells must sum to one")


def factorize_joint(
    variables: tuple[str, ...], order: tuple[str, ...], joint: tuple[float, ...]
) -> tuple[tuple[float, ...], ...]:
    """Return one binary conditional table per variable in causal order."""
    validate_joint(variables, order, joint)
    positions = tuple(variables.index(name) for name in order)
    assignments = binary_assignments(len(variables))
    tables = []
    for depth, position in enumerate(positions):
        table = []
        for prefix in binary_assignments(depth):
            matching = [
                (state, p) for state, p in zip(assignments, joint)
                if tuple(state[i] for i in positions[:depth]) == prefix
            ]
            total = fsum(p for _, p in matching)
            table.append(fsum(p for state, p in matching if state[position]) / total)
        tables.append(tuple(table))
    return tuple(tables)


def factorized_probability(
    assignment: tuple[int, ...], variables: tuple[str, ...],
    order: tuple[str, ...], tables: tuple[tuple[float, ...], ...],
    skip: str | None = None,
) -> float:
    """Evaluate a factorization, optionally omitting an intervention target."""
    if len(assignment) != len(variables):
        raise ValueError("assignment length must match variables")
    assignment_index(assignment)
    probability = 1.0
    prefix_index = 0
    for name, table in zip(order, tables):
        value = assignment[variables.index(name)]
        if name != skip:
            p_one = table[prefix_index]
            probability *= p_one if value else 1.0 - p_one
        prefix_index = 2 * prefix_index + value
    return probability
