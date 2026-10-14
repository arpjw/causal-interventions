"""Post-lock, cross-runtime v3 audit with independent Bayes-risk calculations.

The frozen verifier compares prompt bytes. An assisted prompt embeds exact
floating point action values, whose last bits differ between macOS/Python 3.14
and Linux/Python 3.12. This audit compares those values within 1e-10 while
requiring every nonnumeric public prompt field and model response to match.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path

from causal_interventions.action_v3 import render_action_prompt, summarize_action_runs
from causal_interventions.beliefs import BeliefModel
from causal_interventions.benchmark import read_manifest, validate_manifest
from causal_interventions.policies import one_step_risk
from causal_interventions.prompts import action_id


def equivalent(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, (float, int)):
        return isclose(a, b, rel_tol=0, abs_tol=1e-10)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equivalent(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledger_dir", type=Path)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if manifest["schema_version"] != 2:
        raise ValueError("locked manifest required")
    metadata = json.loads((args.ledger_dir / "metadata.json").read_text())
    if metadata["status"] != "completed" or metadata["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("incomplete or wrong-lock ledger")
    episodes = [e for e in validate_manifest(manifest) if e.split == metadata["split"]]
    rows = [json.loads(line) for line in (args.ledger_dir / "runs.jsonl").read_text().splitlines()]
    if len(rows) != len(episodes) or [row["episode_id"] for row in rows] != [e.episode_id for e in episodes]:
        raise ValueError("ledger count or order mismatch")
    for episode, row in zip(episodes, rows, strict=True):
        expected = render_action_prompt(episode, metadata["condition"] == "assisted")
        marker = "PUBLIC_TASK_JSON\n"
        before, expected_payload = expected.split(marker, 1)
        actual_before, actual_payload = row["prompt"].split(marker, 1)
        if before != actual_before or not equivalent(json.loads(expected_payload), json.loads(actual_payload)):
            raise ValueError("public prompt mismatch")
        try:
            response = json.loads(row["generation"]["text"])
        except ValueError:
            response = None
        model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
        values = {action_id(action): one_step_risk(model, model.prior, action)
                  for action in episode.actions}
        allowed = set(values) | {"STOP"}
        selected = response.get("action_id") if isinstance(response, dict) and set(response) == {"action_id"} else None
        if selected not in allowed:
            selected = None
        if selected != row["selected_action"] or (selected is None) != row["invalid_action"]:
            raise ValueError("response parsing mismatch")
        expected_scores = {
            "model_risk": values.get(selected, model.risk(model.prior)),
            "random_risk": fsum(values.values()) / len(values),
            "oracle_risk": min(model.risk(model.prior), *values.values()),
            "initial_risk": model.risk(model.prior),
        }
        for key, value in expected_scores.items():
            if not isclose(row[key], value, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"independent {key} mismatch")
    calculated = summarize_action_runs(episodes, rows)
    saved = json.loads((args.ledger_dir / "summary.json").read_text())
    if not isclose(calculated["model_minus_random"], saved["model_minus_random"], rel_tol=0, abs_tol=1e-10):
        raise ValueError("summary score mismatch")
    print(json.dumps({"status": "verified", "episodes": len(episodes),
                      "manifest_hash": manifest["manifest_hash"],
                      "ledger_sha256": sha256((args.ledger_dir / "runs.jsonl").read_bytes()).hexdigest(),
                      "model_minus_random": saved["model_minus_random"]}, sort_keys=True))


if __name__ == "__main__":
    main()
