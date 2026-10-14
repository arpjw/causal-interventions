"""Run one v4 condition against the development or frozen test split."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import platform
from time import monotonic

from causal_interventions.action_v4 import (CONDITIONS, action_labels, run_action,
                                             summarize_action_runs)
from causal_interventions.benchmark import read_manifest, validate_manifest, verify_freeze_environment
from causal_interventions.episodes import canonical_json
from causal_interventions.vllm_backend import LocalVLLMBackend

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--action-max-tokens", type=int, default=96)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if (args.max_wall_seconds <= 0 or args.action_max_tokens < 1
            or args.tensor_parallel_size < 1
            or (args.max_episodes is not None and args.max_episodes < 1)):
        parser.error("limits must be positive")
    manifest = read_manifest(args.manifest)
    if manifest["schema_version"] == 2:
        verify_freeze_environment(manifest, ROOT)
        candidate = manifest["candidate"]
    else:
        candidate = manifest
    if candidate["config"]["analysis_version"] != 4:
        parser.error("v4 manifest required")
    if args.split == "test" and manifest["schema_version"] != 2:
        parser.error("test calls require a locked manifest")
    episodes = [episode for episode in validate_manifest(manifest) if episode.split == args.split]
    if args.max_episodes is not None:
        if args.split == "test":
            parser.error("test split must be complete")
        episodes = episodes[:args.max_episodes]
    snapshot = args.model_snapshot.resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        parser.error("model snapshot missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    labels = list(action_labels(episodes[0], args.condition)) + ["STOP"]
    schema = {"type": "object", "additionalProperties": False,
              "required": ["action_id"],
              "properties": {"action_id": {"type": "string", "enum": labels}}}
    metadata = {
        "status": "started", "manifest_hash": manifest["manifest_hash"],
        "snapshot": str(snapshot),
        "model_config_sha256": sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "condition": args.condition, "split": args.split, "backend": "vllm", "device": "cuda",
        "python_version": platform.python_version(), "episodes": len(episodes),
        "max_wall_seconds": args.max_wall_seconds, "action_max_tokens": args.action_max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "maximum_model_calls": len(episodes),
        "maximum_generated_tokens": len(episodes) * args.action_max_tokens,
        "decoding": {"temperature": 0.6 if args.condition == "opaque_thinking" else 0.7,
                     "top_p": 0.95 if args.condition == "opaque_thinking" else 0.8,
                     "top_k": 20, "seed": "sha256(prompt)[:32]",
                     "guided_action_schema": True,
                     "thinking": args.condition == "opaque_thinking"},
        "episode_ids": [episode.episode_id for episode in episodes],
    }
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    started = monotonic()
    print(canonical_json({"status": "loading_model", "episodes": len(episodes)}), flush=True)
    backend = LocalVLLMBackend(snapshot, thinking=args.condition == "opaque_thinking",
                               tensor_parallel_size=args.tensor_parallel_size,
                               json_schema=schema)
    runs = []
    with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for episode in episodes:
            if monotonic() - started >= args.max_wall_seconds:
                break
            row = run_action(episode, backend, args.condition,
                             args.action_max_tokens, allow_locked_test=args.split == "test")
            stream.write(canonical_json(row) + "\n")
            stream.flush()
            runs.append(row)
            print(canonical_json({"completed": len(runs), "episode_id": episode.episode_id,
                                  "selected_label": row["selected_label"],
                                  "selected_action": row["selected_action"],
                                  "invalid_action": row["invalid_action"]}), flush=True)
    metadata["completed"] = len(runs)
    metadata["total_wall_seconds"] = monotonic() - started
    metadata["status"] = "completed" if len(runs) == len(episodes) else "wall_time_cap_reached"
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    if len(runs) == len(episodes):
        report = summarize_action_runs(episodes, runs)
        report["manifest_hash"] = manifest["manifest_hash"]
        report["condition"] = args.condition
        report["run_status"] = metadata["status"]
        (args.output_dir / "summary.json").write_text(canonical_json(report) + "\n")
        print(canonical_json({"status": metadata["status"],
                              "model_minus_random": report["model_minus_random"],
                              "interval_95": report["interval_95"]}), flush=True)
    else:
        print(canonical_json({"status": metadata["status"], "completed": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
