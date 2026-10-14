"""Apply the frozen v5 paired decision rule to five independently verified ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_interventions.action_v3 import compare_action_runs
from causal_interventions.action_v5 import CONDITIONS
from causal_interventions.episodes import canonical_json
from scripts.verify_action_v5 import verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledgers", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_condition = {}
    episodes = None
    reference_ids = None
    for ledger in args.ledgers:
        current, rows = verify(args.manifest, ledger)
        condition = json.loads((ledger / "metadata.json").read_text())["condition"]
        ids = [e.episode_id for e in current]
        if reference_ids is None:
            reference_ids, episodes = ids, current
        elif ids != reference_ids:
            raise ValueError("conditions cover different cases")
        if condition in by_condition:
            raise ValueError("duplicate condition")
        by_condition[condition] = rows
    if set(by_condition) != set(CONDITIONS) or episodes is None:
        raise ValueError("all five registered conditions are required")
    primary = compare_action_runs(episodes, by_condition["q235_short_raw_stages"],
                                  by_condition["q235_long_raw_stages"])
    primary["criterion_met"] = (primary["interval_95"][1] < 0
                                and primary["second_minus_first"] <= -0.005)
    contrasts = {
        "q235_likelihood_minus_short_raw": compare_action_runs(
            episodes, by_condition["q235_short_raw_stages"],
            by_condition["q235_likelihood_stages"]),
        "q32_calculator_minus_raw": compare_action_runs(
            episodes, by_condition["q32_raw_stages"],
            by_condition["q32_calculator_agent"]),
    }
    output = {"status": "completed", "primary_long_minus_short": primary,
              "descriptive_contrasts": contrasts}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(output) + "\n")
    print(canonical_json(output))


if __name__ == "__main__":
    main()
