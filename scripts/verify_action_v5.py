"""Replay a complete v5 ledger and independently recompute its public-input risks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum, isclose
from pathlib import Path
from typing import Any

from causal_interventions.action_v5 import (CONDITIONS, STAGE_CONDITIONS, action_labels,
                                             render_stage_prompt, render_tool_prompt,
                                             score_stage, summarize)
from causal_interventions.beliefs import BeliefModel
from causal_interventions.benchmark import read_manifest, validate_manifest
from causal_interventions.episodes import Episode, canonical_json
from causal_interventions.policies import one_step_risk
from causal_interventions.prompts import action_id
from causal_interventions.runner import Generation
from scripts.verify_action_v4 import equivalent


def prompt_equivalent(expected: str, actual: str, *, tool: bool = False) -> bool:
    marker = "PUBLIC_TASK_JSON\n"
    expected_before, expected_body = expected.split(marker, 1)
    actual_before, actual_body = actual.split(marker, 1)
    if expected_before != actual_before:
        return False
    if tool:
        boundary = "\n\nTOOL_TRANSCRIPT_JSON\n"
        expected_task, expected_transcript = expected_body.split(boundary, 1)
        actual_task, actual_transcript = actual_body.split(boundary, 1)
        return (equivalent(json.loads(expected_task), json.loads(actual_task))
                and equivalent(json.loads(expected_transcript), json.loads(actual_transcript)))
    return equivalent(json.loads(expected_body), json.loads(actual_body))


def independent_risks(episode: Episode) -> tuple[dict[str, float], float, float, float]:
    model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
    labels = action_labels(episode)
    by_semantic = {action_id(action): one_step_risk(model, model.prior, action)
                   for action in episode.actions}
    by_label = {label: by_semantic[semantic] for label, semantic in labels.items()}
    random = fsum(by_label.values()) / len(by_label)
    initial = model.risk(model.prior)
    return by_label, random, min(initial, *by_label.values()), initial


def verify(manifest_path: Path, ledger_dir: Path) -> tuple[list[Episode], list[dict[str, Any]]]:
    manifest = read_manifest(manifest_path)
    if (manifest["schema_version"] != 2 or manifest["candidate"]["config"]["seed"] != 827419):
        raise ValueError("locked v5 manifest required")
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
        if (row["condition"] != condition or row["parameter_id"] != episode.parameter_id
                or row["split"] != episode.split or row["stratum"] != episode.stratum):
            raise ValueError("case identity mismatch")
        risks, random, oracle, initial = independent_risks(episode)
        if condition in STAGE_CONDITIONS:
            if not prompt_equivalent(render_stage_prompt(episode, condition), row["prompt"]):
                raise ValueError("stage prompt mismatch")
            generation = Generation(**row["generation"])
            expected = score_stage(episode, condition, generation, allow_locked_test=True)
            if not equivalent({k: v for k, v in expected.items() if k != "prompt"},
                              {key: row[key] for key in expected if key != "prompt"}):
                raise ValueError("stage parsing or metric mismatch")
        else:
            transcript: list[dict[str, Any]] = []
            final = None
            for index, call in enumerate(row["calls"]):
                if not prompt_equivalent(render_tool_prompt(episode, transcript),
                                         call["prompt"], tool=True):
                    raise ValueError("tool prompt or transcript mismatch")
                try:
                    response = json.loads(call["generation"]["text"])
                except (TypeError, ValueError):
                    response = None
                if (not isinstance(response, dict) or set(response) != {"operation", "action_id"}
                        or response["action_id"] not in (*"ABC", "STOP")
                        or response["operation"] not in ("evaluate", "final")):
                    if index != len(row["calls"]) - 1:
                        raise ValueError("calls continue after invalid response")
                    break
                if response["operation"] == "final":
                    final = response["action_id"]
                    if index != len(row["calls"]) - 1:
                        raise ValueError("calls continue after final response")
                    break
                if index >= 3 or response["action_id"] == "STOP":
                    if index != len(row["calls"]) - 1:
                        raise ValueError("calls continue after invalid tool request")
                    break
                label = response["action_id"]
                transcript.append({"operation": "evaluate_action", "action_id": label,
                                   "expected_risk_nats": risks[label]})
            if (len(row["calls"]) > 4 or not equivalent(row["tool_transcript"], transcript)
                    or row["selected_label"] != final
                    or row["distinct_tool_actions"] != len({t["action_id"] for t in transcript})):
                raise ValueError("tool replay mismatch")
        label = row["selected_label"]
        selected = action_labels(episode).get(label, "STOP") if label is not None else None
        expected_scores = {"model_risk": risks.get(label, initial), "random_risk": random,
                           "oracle_risk": oracle, "initial_risk": initial}
        if row["selected_action"] != selected or row["invalid_action"] != (selected is None):
            raise ValueError("selected action mismatch")
        for key, value in expected_scores.items():
            if not isclose(row[key], value, rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"independent {key} mismatch: {episode.episode_id}")
    calculated = summarize(episodes, rows)
    saved = json.loads((ledger_dir / "summary.json").read_text())
    for key in ("model_minus_random", "interval_95", "mean_risks", "stage_mae",
                "self_consistent_actions", "mean_distinct_tool_actions", "all_three_tool_actions"):
        if key in calculated and not equivalent(calculated[key], saved.get(key)):
            raise ValueError(f"summary mismatch: {key}")
    digest = sha256((ledger_dir / "runs.jsonl").read_bytes()).hexdigest()
    print(canonical_json({"status": "verified", "condition": condition, "episodes": len(episodes),
                          "ledger_sha256": digest, "model_minus_random": saved["model_minus_random"]}))
    return episodes, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledger_dir", type=Path)
    args = parser.parse_args()
    verify(args.manifest, args.ledger_dir)


if __name__ == "__main__":
    main()
