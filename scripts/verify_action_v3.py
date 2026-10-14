"""Replay a complete v3 ledger with the separate belief and policy implementation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isclose, fsum
from pathlib import Path

from causal_interventions.action_v3 import compare_action_runs, parse_action, render_action_prompt, summarize_action_runs
from causal_interventions.beliefs import BeliefModel
from causal_interventions.benchmark import read_manifest, validate_manifest
from causal_interventions.episodes import canonical_json
from causal_interventions.policies import one_step_risk
from causal_interventions.prompts import action_id


def verify(manifest_path: Path, output_dir: Path) -> tuple[list, list[dict]]:
    manifest = read_manifest(manifest_path)
    if manifest["schema_version"] != 2:
        raise ValueError("locked manifest required")
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata["status"] != "completed" or metadata["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("run is incomplete or bound to another lock")
    episodes = [episode for episode in validate_manifest(manifest) if episode.split == metadata["split"]]
    if len(episodes) != metadata["episodes"]:
        raise ValueError("episode count mismatch")
    rows = [json.loads(line) for line in (output_dir / "runs.jsonl").read_text().splitlines()]
    if len(rows) != len(episodes) or [r["episode_id"] for r in rows] != [e.episode_id for e in episodes]:
        raise ValueError("ledger is incomplete or out of order")
    for episode, row in zip(episodes, rows, strict=True):
        if row["prompt"] != render_action_prompt(episode, metadata["condition"] == "assisted"):
            raise ValueError("recorded prompt differs from public task")
        selected, invalid = parse_action(row["generation"]["text"], episode)
        if selected != row["selected_action"] or invalid != row["invalid_action"]:
            raise ValueError("response parsing mismatch")
        model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
        values = {action_id(action): one_step_risk(model, model.prior, action)
                  for action in episode.actions}
        expected = {
            "model_risk": values.get(selected, model.risk(model.prior)),
            "random_risk": fsum(values.values()) / len(values),
            "oracle_risk": min(model.risk(model.prior), *values.values()),
            "initial_risk": model.risk(model.prior),
        }
        for name, value in expected.items():
            if not isclose(row[name], value, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"independent {name} mismatch for {episode.episode_id}")
    report = summarize_action_runs(episodes, rows)
    saved = json.loads((output_dir / "summary.json").read_text())
    if saved["model_minus_random"] != report["model_minus_random"] or saved["interval_95"] != report["interval_95"]:
        raise ValueError("summary disagrees with replay")
    print(canonical_json({"status": "verified", "episodes": len(episodes),
                          "manifest_hash": manifest["manifest_hash"],
                          "ledger_sha256": sha256((output_dir / "runs.jsonl").read_bytes()).hexdigest(),
                          "model_minus_random": report["model_minus_random"]}))
    return episodes, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    episodes, rows = verify(args.manifest, args.output_dir)
    if args.compare_to:
        other_episodes, other_rows = verify(args.manifest, args.compare_to)
        if [e.episode_id for e in other_episodes] != [e.episode_id for e in episodes]:
            raise ValueError("comparison episode sets differ")
        print(canonical_json(compare_action_runs(episodes, rows, other_rows)))


if __name__ == "__main__":
    main()
