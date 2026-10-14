"""Independently check the one-step query-information identity on locked cases."""

from __future__ import annotations

import argparse
from math import fsum, log
from pathlib import Path

from causal_interventions.beliefs import BeliefModel
from causal_interventions.benchmark import read_manifest, validate_manifest
from causal_interventions.policies import one_step_risk


def query_experiment_information(model: BeliefModel, action) -> float:
    """Compute I(Y;O_a) directly from the joint query/experiment table."""
    if len(model.queries) != 1:
        raise ValueError("one binary query required")
    q = [row[0] for row in model.query_probabilities]
    likelihood = model.likelihoods[action]
    outcome_marginal = model.outcomes(model.prior, action)
    query_marginal = fsum(b * value for b, value in zip(model.prior, q))
    total = 0.0
    for y in (0, 1):
        py = query_marginal if y else 1 - query_marginal
        for o, po in enumerate(outcome_marginal):
            joint = fsum(b * (value if y else 1 - value) * row[o]
                         for b, value, row in zip(model.prior, q, likelihood))
            if joint > 0:
                total += joint * log(joint / (py * po))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    cases = [e for e in validate_manifest(read_manifest(args.manifest)) if e.split == "test"]
    errors = []
    for episode in cases:
        model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
        initial = model.risk(model.prior)
        for action in episode.actions:
            errors.append(abs(initial - one_step_risk(model, model.prior, action)
                              - query_experiment_information(model, action)))
    print({"cases": len(cases), "action_checks": len(errors), "max_absolute_error": max(errors)})
    if max(errors) > 1e-10:
        raise SystemExit("information identity failed")


if __name__ == "__main__":
    main()
