"""Combine verified v4 condition ledgers into paired mechanism contrasts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from causal_interventions.action_v3 import compare_action_runs
from causal_interventions.episodes import canonical_json
from scripts.verify_action_v4 import verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledgers", nargs="+", type=Path)
    args = parser.parse_args()
    by_condition = {}
    reference_ids = None
    episodes = None
    for ledger in args.ledgers:
        current_episodes, rows = verify(args.manifest, ledger)
        condition = json.loads((ledger / "metadata.json").read_text())["condition"]
        ids = [episode.episode_id for episode in current_episodes]
        if reference_ids is None:
            reference_ids, episodes = ids, current_episodes
        elif ids != reference_ids:
            raise ValueError("conditions use different episode sets")
        if condition in by_condition:
            raise ValueError("duplicate condition")
        by_condition[condition] = rows
    if episodes is None:
        raise ValueError("at least one ledger required")
    report = {}
    for condition, rows in by_condition.items():
        report[condition] = {
            "presented_label_counts": dict(Counter(row["selected_label"] or "INVALID" for row in rows)),
            "semantic_action_counts": dict(Counter(row["selected_action"] or "INVALID" for row in rows)),
            "per_stratum": {stratum: {
                "episodes": len(group),
                "model_minus_random": sum(row["model_risk"] - row["random_risk"] for row in group) / len(group),
                "optimal_choices": sum(abs(row["model_risk"] - row["oracle_risk"]) < 1e-10 for row in group),
            } for stratum in sorted({row["stratum"] for row in rows})
              for group in [[row for row in rows if row["stratum"] == stratum]]},
        }
    contrasts = (("literal_raw", "opaque_raw"),
                 ("opaque_raw", "opaque_likelihood"),
                 ("opaque_likelihood", "opaque_values"),
                 ("opaque_raw", "opaque_thinking"))
    report["paired_contrasts"] = {
        f"{second}_minus_{first}": compare_action_runs(episodes, by_condition[first], by_condition[second])
        for first, second in contrasts if first in by_condition and second in by_condition
    }
    print(canonical_json(report))


if __name__ == "__main__":
    main()
