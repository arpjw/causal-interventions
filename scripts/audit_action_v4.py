"""Post hoc label-bias diagnostics for complete, verified v4 test ledgers."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from random import Random
from typing import Any

from causal_interventions.action_v3 import action_values
from causal_interventions.action_v4 import action_labels
from causal_interventions.episodes import Episode, canonical_json
from scripts.verify_action_v4 import verify


def interval(values: list[float], *, seed: int = 161803, samples: int = 4000) -> list[float]:
    if len(values) < 2:
        raise ValueError("at least two independent values required")
    rng = Random(seed)
    means = sorted(fsum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(samples))

    def quantile(p: float) -> float:
        index = (len(means) - 1) * p
        low = int(index)
        weight = index - low
        return means[low] * (1 - weight) + means[min(low + 1, len(means) - 1)] * weight

    return [quantile(0.025), quantile(0.975)]


def condition_audit(episodes: list[Episode], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [(episode, row) for episode, row in zip(episodes, rows, strict=True)
                if episode.stratum.startswith("informative")]
    if len({episode.parameter_id for episode, _ in selected}) != len(selected):
        raise ValueError("audit expects independent informative parameter draws")
    chosen = Counter()
    optimal = Counter()
    contingency: dict[str, Counter[str]] = {label: Counter() for label in "ABC"}
    model_minus_fixed: dict[str, list[float]] = {label: [] for label in "ABC"}
    fixed_minus_random: dict[str, list[float]] = {label: [] for label in "ABC"}
    optimal_count = 0
    for episode, row in selected:
        labels = action_labels(episode, "opaque_raw")
        risks = action_values(episode)
        by_label = {label: risks[action] for label, action in labels.items()}
        oracle = min(by_label, key=by_label.get)
        label = row["selected_label"] or "INVALID"
        chosen[label] += 1
        optimal[oracle] += 1
        contingency[oracle][label] += 1
        optimal_count += abs(row["model_risk"] - by_label[oracle]) < 1e-10
        for fixed_label in "ABC":
            model_minus_fixed[fixed_label].append(row["model_risk"] - by_label[fixed_label])
            fixed_minus_random[fixed_label].append(by_label[fixed_label] - row["random_risk"])
    result: dict[str, Any] = {
        "informative_episodes": len(selected),
        "selected_label_counts": dict(chosen),
        "optimal_label_counts": dict(optimal),
        "optimal_choice_count": optimal_count,
        "optimal_by_oracle_label_and_choice": {key: dict(value) for key, value in contingency.items()},
        "fixed_label_baselines": {},
    }
    for label in "ABC":
        model_diff = model_minus_fixed[label]
        fixed_diff = fixed_minus_random[label]
        result["fixed_label_baselines"][label] = {
            "model_minus_fixed_risk": fsum(model_diff) / len(model_diff),
            "model_minus_fixed_interval_95": interval(model_diff),
            "fixed_minus_random_risk": fsum(fixed_diff) / len(fixed_diff),
            "fixed_optimal_count": optimal[label],
            "model_selects_label_when_optimal": contingency[label][label],
            "model_selects_label_when_not_optimal": chosen[label] - contingency[label][label],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("raw_32b", type=Path)
    parser.add_argument("thinking_235b", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes32, rows32 = verify(args.manifest, args.raw_32b)
    episodes235, rows235 = verify(args.manifest, args.thinking_235b)
    if [e.episode_id for e in episodes32] != [e.episode_id for e in episodes235]:
        raise ValueError("ledgers must cover identical cases")
    chosen32 = [r["selected_label"] for r in rows32]
    chosen235 = [r["selected_label"] for r in rows235]
    informative = [e.stratum.startswith("informative") for e in episodes32]
    report = {
        "status": "post_hoc_exploratory",
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "ledger_sha256": {"opaque_raw_32b": sha256((args.raw_32b / "runs.jsonl").read_bytes()).hexdigest(),
                          "opaque_thinking_235b": sha256((args.thinking_235b / "runs.jsonl").read_bytes()).hexdigest()},
        "opaque_raw_32b": condition_audit(episodes32, rows32),
        "opaque_thinking_235b": condition_audit(episodes235, rows235),
        "cross_model": {
            "same_presented_label_all": sum(a == b for a, b in zip(chosen32, chosen235, strict=True)),
            "same_presented_label_informative": sum(a == b for a, b, keep in
                                                     zip(chosen32, chosen235, informative, strict=True) if keep),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n")
    print(canonical_json(report))


if __name__ == "__main__":
    main()
