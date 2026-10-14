"""Evaluate every test episode from a verified locked manifest on CUDA."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import platform
from time import monotonic

from causal_interventions.analysis import analyze_paired
from causal_interventions.analysis_v2 import analyze_v2
from causal_interventions.benchmark import read_manifest, validate_manifest, verify_freeze_environment
from causal_interventions.episodes import canonical_json
from causal_interventions.runner import run_episode
from causal_interventions.vllm_backend import LocalVLLMBackend


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--action-max-tokens", type=int, default=96)
    parser.add_argument("--forecast-max-tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.max_wall_seconds <= 0 or args.action_max_tokens < 1 or args.forecast_max_tokens < 1:
        parser.error("wall time and token limits must be positive")
    manifest = read_manifest(args.manifest)
    if manifest.get("schema_version") != 2 or manifest.get("status") != "locked":
        parser.error("a locked manifest is required")
    verify_freeze_environment(manifest, ROOT)
    episodes = [episode for episode in validate_manifest(manifest) if episode.split == "test"]
    version = manifest["candidate"]["analysis"].get("primary_comparison") == "model_action_bayes_minus_random_on_informative_strata"
    snapshot = args.model_snapshot.resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        parser.error("model snapshot is missing")
    model_id = f"vllm-local:{snapshot.parent.parent.name}:{snapshot.name}:{'thinking' if args.thinking else 'direct'}" if version else f"vllm-local:{snapshot.parent.parent.name}:{snapshot.name}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "started", "manifest_hash": manifest["manifest_hash"],
        "model_id": model_id, "snapshot": str(snapshot),
        "model_config_sha256": sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "backend": "vllm", "device": "cuda", "python_version": platform.python_version(),
        "thinking": args.thinking, "analysis_version": 2 if version else 1,
        "thinking_stage_token_cap": 384 if args.thinking else None,
        "thinking_final_json_constrained": args.thinking,
        "test_episode_count": len(episodes), "wall_seconds_cap": args.max_wall_seconds,
        "maximum_model_calls": sum(episode.budget + 1 for episode in episodes),
        "maximum_backend_generate_calls": sum(episode.budget + 1 for episode in episodes) * (2 if args.thinking else 1),
        "maximum_generated_tokens": sum(
            episode.budget * args.action_max_tokens + args.forecast_max_tokens
            for episode in episodes
        ),
        "action_max_tokens": args.action_max_tokens,
        "forecast_max_tokens": args.forecast_max_tokens,
        "decoding": {"temperature": 0.6 if args.thinking else 0.7,
                     "top_p": 0.95 if args.thinking else 0.8, "top_k": 20,
                     "seed": "sha256(prompt)[:32]", "json_object": True,
                     "trust_remote_code": False, "local_files_only": True},
        "episode_ids": [episode.episode_id for episode in episodes],
    }
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    print(canonical_json({"status": "loading_model", "test_episodes": len(episodes)}), flush=True)
    started = monotonic()
    backend = LocalVLLMBackend(snapshot, thinking=args.thinking)
    runs = []
    with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for index, episode in enumerate(episodes):
            if monotonic() - started >= args.max_wall_seconds:
                break
            run = run_episode(episode, backend, model_id,
                              args.action_max_tokens, args.forecast_max_tokens,
                              allow_locked_test=True, score_actions=version,
                              prompt_version="primary-v2" if version else "primary-v1")
            stream.write(canonical_json(run.record()) + "\n")
            stream.flush()
            runs.append(run)
            print(canonical_json({"completed": index + 1,
                                  "episode_id": episode.episode_id,
                                  "elapsed_seconds": run.trace["elapsed_seconds"],
                                  "parse_failures": run.trace["parse_failures"]}), flush=True)
    metadata["status"] = "completed" if len(runs) == len(episodes) else "wall_time_cap_reached"
    metadata["completed"] = len(runs)
    metadata["total_wall_seconds"] = monotonic() - started
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    if runs:
        references = [run.reference for run in runs]
        results = [result for run in runs for result in run.results]
        if version:
            informative_ids = {e.episode_id for e in episodes[:len(runs)]
                               if e.stratum.startswith("informative")}
            report = analyze_v2(references, results, model_id, informative_ids)
            primary = report["primary_action_choice"]
        else:
            report = analyze_paired(references, results, model_id)
            primary = report
        report["run_status"] = metadata["status"]
        report["manifest_hash"] = manifest["manifest_hash"]
        (args.output_dir / "summary.json").write_text(canonical_json(report) + "\n")
        print(canonical_json({"status": metadata["status"], "completed": len(runs),
                              "model_minus_random": primary["paired_model_minus_random"],
                              "opportunity_gap": primary["opportunity_gap"],
                              "parse_failures": sum(r.trace["parse_failures"] for r in runs)}), flush=True)


if __name__ == "__main__":
    main()
