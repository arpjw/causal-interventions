"""Independent finite-table replay of a v2 locked result ledger."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import log
from pathlib import Path
from statistics import mean

STATES = tuple((x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1))
NAMES = ("X", "Y", "Z")


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def intervened(joint: list[float], order: list[str], target: str, value: int) -> list[float]:
    output = []
    for state in STATES:
        if state[NAMES.index(target)] != value:
            output.append(0.0)
            continue
        mass = 1.0
        prefix: dict[str, int] = {}
        for variable in order:
            if variable != target:
                denom = sum(joint[i] for i, candidate in enumerate(STATES)
                            if all(candidate[NAMES.index(k)] == v for k, v in prefix.items()))
                numer = sum(joint[i] for i, candidate in enumerate(STATES)
                            if all(candidate[NAMES.index(k)] == v for k, v in prefix.items())
                            and candidate[NAMES.index(variable)] == state[NAMES.index(variable)])
                mass *= numer / denom
            prefix[variable] = state[NAMES.index(variable)]
        output.append(mass)
    if abs(sum(output) - 1) > 1e-10:
        raise AssertionError("intervention fails normalization")
    return output


def loss(truth: float, forecast: float) -> float:
    forecast = min(1 - 1e-6, max(1e-6, forecast))
    return sum(p * log(p / q) for p, q in ((truth, forecast), (1 - truth, 1 - forecast)) if p > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    lock = json.loads(args.manifest.read_text())
    assert digest({k: v for k, v in lock.items() if k != "manifest_hash"}) == lock["manifest_hash"]
    repository = Path(__file__).resolve().parents[1]
    for relative, expected_hash in lock["freeze"]["source_hashes"].items():
        assert sha256((repository / relative).read_bytes()).hexdigest() == expected_hash, relative
    episodes = {r["episode_id"]: r for r in lock["candidate"]["episodes"] if r["split"] == "test"}
    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    report = json.loads((args.run_dir / "summary.json").read_text())
    runs = [json.loads(line) for line in (args.run_dir / "runs.jsonl").read_text().splitlines()]
    assert metadata["status"] == "completed" and metadata["completed"] == len(episodes) == len(runs) == 150
    assert metadata["manifest_hash"] == lock["manifest_hash"]
    assert {r["reference"]["episode_id"] for r in runs} == set(episodes)
    model_id = metadata["model_id"]
    groups: dict[str, list[tuple[float, float, float]]] = {"informative": [], "near_tie": []}
    failures = 0
    max_generated = 0
    for run in runs:
        eid = run["reference"]["episode_id"]
        episode = episodes[eid]
        joint = episode["joint"]
        orders = episode["orders"]
        query = episode["queries"][0]
        query_target = query["intervention"]["target"]
        query_value = query["intervention"]["value"]
        outcome_index = NAMES.index(query["outcome"])
        world_probs = [sum(p for state, p in zip(STATES, intervened(joint, order, query_target, query_value))
                           if state[outcome_index] == query["outcome_value"]) for order in orders]
        truth = world_probs[episode["hidden_world"]]
        assert abs(truth - run["reference"]["query_truths"][0]) < 1e-10
        posterior = episode["prior"][:]
        for step in run["trace"]["history"]:
            if step["action"] is None:
                assert step["outcome"] is None
                continue
            target, value = step["action"]["target"], step["action"]["value"]
            idx = STATES.index(tuple(step["outcome"]))
            likelihood = [intervened(joint, order, target, value)[idx] for order in orders]
            weighted = [p * l for p, l in zip(posterior, likelihood)]
            posterior = [v / sum(weighted) for v in weighted]
        expected = sum(p * q for p, q in zip(posterior, world_probs))
        rows = {r["policy_id"]: r for r in run["results"]}
        assert set(rows) == {model_id, model_id + ":action_bayes", "random", "exact"}
        actual = rows[model_id + ":action_bayes"]["submitted_probabilities"][0]
        assert abs(actual - expected) < 1e-10, (eid, actual, expected)
        scores = tuple(loss(truth, rows[p]["submitted_probabilities"][0])
                       for p in (model_id + ":action_bayes", "random", "exact"))
        groups["informative" if episode["stratum"].startswith("informative") else "near_tie"].append(scores)
        failures += run["trace"]["parse_failures"]
        max_generated = max(max_generated, *(c["output_tokens"] for c in run["trace"]["calls"]))
    informative = groups["informative"]
    assert len(informative) == 100 and len(groups["near_tie"]) == 50
    means = [mean(row[i] for row in informative) for i in range(3)]
    primary = report["primary_action_choice"]
    assert abs(primary["paired_model_minus_random"] - (means[0] - means[1])) < 1e-10
    for key, value in zip((model_id + ":action_bayes", "random", "exact"), means):
        assert abs(primary["mean_losses"][key] - value) < 1e-10
    print(json.dumps({"lock_hash": lock["manifest_hash"], "episodes": len(runs),
                      "informative": len(informative), "near_tie": len(groups["near_tie"]),
                      "mean_losses": dict(zip(("model_action_bayes", "random", "exact"), means)),
                      "paired_model_minus_random": means[0] - means[1],
                      "parse_failures": failures, "max_output_tokens_per_call": max_generated},
                     sort_keys=True))


if __name__ == "__main__":
    main()
