"""Independently replay a complete v4 ledger across Python/libm runtimes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path

from causal_interventions.action_v4 import (CONDITIONS, action_labels,
                                             render_action_prompt,
                                             summarize_action_runs)
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


def verify(manifest_path: Path, ledger_dir: Path) -> tuple[list, list[dict]]:
    manifest = read_manifest(manifest_path)
    if manifest["schema_version"] != 2 or manifest["candidate"]["config"]["analysis_version"] != 4:
        raise ValueError("locked v4 manifest required")
    metadata = json.loads((ledger_dir / "metadata.json").read_text())
    condition = metadata["condition"]
    if (condition not in CONDITIONS or metadata["status"] != "completed"
            or metadata["manifest_hash"] != manifest["manifest_hash"]):
        raise ValueError("incomplete, unknown-condition, or wrong-lock ledger")
    episodes = [e for e in validate_manifest(manifest) if e.split == metadata["split"]]
    rows = [json.loads(line) for line in (ledger_dir / "runs.jsonl").read_text().splitlines()]
    if len(rows) != len(episodes) or [r["episode_id"] for r in rows] != [e.episode_id for e in episodes]:
        raise ValueError("ledger count or order mismatch")
    for episode, row in zip(episodes, rows, strict=True):
        if row["condition"] != condition or row["parameter_id"] != episode.parameter_id:
            raise ValueError("condition or parameter mismatch")
        marker = "PUBLIC_TASK_JSON\n"
        before, expected_payload = render_action_prompt(episode, condition).split(marker, 1)
        actual_before, actual_payload = row["prompt"].split(marker, 1)
        if before != actual_before or not equivalent(json.loads(expected_payload), json.loads(actual_payload)):
            raise ValueError("public prompt mismatch")
        try:
            response = json.loads(row["generation"]["text"].strip())
        except (TypeError, ValueError):
            response = None
        labels = action_labels(episode, condition)
        selected_label = (response.get("action_id") if isinstance(response, dict)
                          and set(response) == {"action_id"} else None)
        if not isinstance(selected_label, str) or selected_label not in (*labels, "STOP"):
            selected_label = None
        selected = labels.get(selected_label, "STOP") if selected_label is not None else None
        if (selected_label != row["selected_label"] or selected != row["selected_action"]
                or (selected is None) != row["invalid_action"]):
            raise ValueError("response parsing mismatch")
        model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
        values = {action_id(action): one_step_risk(model, model.prior, action)
                  for action in episode.actions}
        expected_scores = {
            "model_risk": values.get(selected, model.risk(model.prior)),
            "random_risk": fsum(values.values()) / len(values),
            "oracle_risk": min(model.risk(model.prior), *values.values()),
            "initial_risk": model.risk(model.prior),
        }
        for name, value in expected_scores.items():
            if not isclose(row[name], value, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"independent {name} mismatch for {episode.episode_id}")
    calculated = summarize_action_runs(episodes, rows)
    saved = json.loads((ledger_dir / "summary.json").read_text())
    if (not isclose(calculated["model_minus_random"], saved["model_minus_random"],
                    rel_tol=0, abs_tol=1e-10)
            or not equivalent(calculated["interval_95"], saved["interval_95"])):
        raise ValueError("summary disagrees with replay")
    print(json.dumps({"status": "verified", "condition": condition,
                      "episodes": len(episodes), "manifest_hash": manifest["manifest_hash"],
                      "ledger_sha256": sha256((ledger_dir / "runs.jsonl").read_bytes()).hexdigest(),
                      "model_minus_random": saved["model_minus_random"]}, sort_keys=True))
    return episodes, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledger_dir", type=Path)
    args = parser.parse_args()
    verify(args.manifest, args.ledger_dir)


if __name__ == "__main__":
    main()
