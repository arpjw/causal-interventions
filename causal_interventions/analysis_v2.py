"""Prespecified action-choice analysis for benchmark v2."""

from __future__ import annotations

from typing import Sequence

from .analysis import EpisodeResult, EpisodeTruth, analyze_paired


def analyze_v2(reference: Sequence[EpisodeTruth], results: Sequence[EpisodeResult],
               model_id: str, informative_ids: set[str],
               bootstrap_samples: int = 2000) -> dict[str, object]:
    if not informative_ids or not informative_ids <= {row.episode_id for row in reference}:
        raise ValueError("informative episode IDs must be a nonempty subset of reference")
    primary_id = f"{model_id}:action_bayes"
    primary_rows = [row for row in results if row.policy_id == primary_id]
    if {row.episode_id for row in primary_rows} != {row.episode_id for row in reference}:
        raise ValueError("action-conditioned Bayesian rows must be complete")
    allowed_primary = {primary_id, "random", "exact"}
    selected_reference = [row for row in reference if row.episode_id in informative_ids]
    primary = analyze_paired(
        selected_reference,
        [row for row in results if row.episode_id in informative_ids
         and row.policy_id in allowed_primary],
        primary_id, bootstrap_samples=bootstrap_samples, minimum_gap=0.01,
    )
    secondary = analyze_paired(
        reference, [row for row in results if row.policy_id in {model_id, "random", "exact"}],
        model_id, bootstrap_samples=bootstrap_samples,
    )
    primary["contract"].update(
        schema_version=2,
        primary_metric="exact_bayesian_forecast_after_model_actions_excess_log_loss_nats",
        primary_comparison="model_action_bayes_minus_random_on_informative_strata",
        informative_episode_only=True,
    )
    near_tie_ids = {row.episode_id for row in reference} - informative_ids
    near_tie = None
    if near_tie_ids:
        near_tie = analyze_paired(
            [row for row in reference if row.episode_id in near_tie_ids],
            [row for row in results if row.episode_id in near_tie_ids
             and row.policy_id in allowed_primary],
            primary_id, bootstrap_samples=bootstrap_samples, minimum_gap=0.01,
        )
    return {"primary_action_choice": primary, "secondary_submitted_forecast": secondary,
            "secondary_near_tie_action": near_tie}
