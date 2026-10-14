"""Run a capped, offline, development-only model pilot."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import platform
from time import monotonic

from causal_interventions.analysis import analyze_paired
from causal_interventions.analysis_v2 import analyze_v2
from causal_interventions.benchmark import read_manifest, validate_manifest
from causal_interventions.episodes import canonical_json
from causal_interventions.hf_backend import LocalHFBackend
from causal_interventions.mlx_backend import LocalMLXBackend
from causal_interventions.vllm_backend import LocalVLLMBackend
from causal_interventions.runner import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--backend", choices=("hf", "mlx", "vllm"), default="hf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    parser.add_argument("--action-max-tokens", type=int, default=96)
    parser.add_argument("--forecast-max-tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_wall_seconds <= 0:
        parser.error("episode and wall-time caps must be positive")
    manifest = read_manifest(args.manifest)
    if manifest["schema_version"] != 1 or manifest["status"] != "candidate_unfrozen":
        parser.error("development pilots require an unfrozen candidate manifest")
    episodes = [e for e in validate_manifest(manifest) if e.split == "development"]
    version = manifest["analysis"].get("primary_comparison") == "model_action_bayes_minus_random_on_informative_strata"
    if args.thinking and not version:
        parser.error("thinking pilot requires benchmark v2")
    if args.episodes > len(episodes):
        parser.error("requested more episodes than the development split contains")
    # Round-robin selection gives a short pilot different problem settings.
    by_stratum = {}
    for episode in episodes:
        by_stratum.setdefault(episode.stratum, []).append(episode)
    selected = []
    for offset in range(max(map(len, by_stratum.values()))):
        for group in by_stratum.values():
            if offset < len(group):
                selected.append(group[offset])
    selected = selected[:args.episodes]
    snapshot = args.model_snapshot.resolve()
    model_id = f"{args.backend}-local:{snapshot.parent.parent.name}:{snapshot.name}:{'thinking' if args.thinking else 'direct'}"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "started", "manifest_hash": manifest["manifest_hash"],
        "model_id": model_id, "snapshot": str(snapshot),
        "device": "cuda" if args.backend == "vllm" else "apple-gpu",
        "backend": args.backend,
        "thinking": args.thinking, "analysis_version": 2 if version else 1,
        "thinking_stage_token_cap": 384 if args.thinking else None,
        "thinking_final_json_constrained": args.thinking,
        "episode_cap": args.episodes, "wall_seconds_cap": args.max_wall_seconds,
        "action_max_tokens": args.action_max_tokens,
        "forecast_max_tokens": args.forecast_max_tokens,
        "episode_ids": [e.episode_id for e in selected],
        "maximum_model_calls": sum(e.budget + 1 for e in selected),
        "maximum_backend_generate_calls": sum(e.budget + 1 for e in selected) * (2 if args.thinking else 1),
        "maximum_generated_tokens": sum(
            e.budget * args.action_max_tokens + args.forecast_max_tokens
            for e in selected
        ),
        "python_version": platform.python_version(),
        "decoding": {"temperature": (0.6 if args.thinking else 0.7) if args.backend == "vllm" else 0.0,
                     "top_p": (0.95 if args.thinking else 0.8) if args.backend == "vllm" else None,
                     "top_k": 20 if args.backend == "vllm" else None,
                     "json_object": args.backend == "vllm",
                     "seed": "sha256(prompt)[:32]" if args.backend == "vllm" else None,
                     "dtype": "auto" if args.backend == "vllm" else ("float16" if args.backend == "hf" else "4bit"),
                     "local_files_only": True,
                     "trust_remote_code": False},
        "source_sha256": {
            "scripts/run_development.py": sha256(Path(__file__).read_bytes()).hexdigest(),
            "causal_interventions/runner.py": sha256(
                (Path(__file__).resolve().parents[1] / "causal_interventions" / "runner.py").read_bytes()
            ).hexdigest(),
            "causal_interventions/hf_backend.py": sha256(
                (Path(__file__).resolve().parents[1] / "causal_interventions" / "hf_backend.py").read_bytes()
            ).hexdigest(),
            "causal_interventions/mlx_backend.py": sha256(
                (Path(__file__).resolve().parents[1] / "causal_interventions" / "mlx_backend.py").read_bytes()
            ).hexdigest(),
            "causal_interventions/vllm_backend.py": sha256(
                (Path(__file__).resolve().parents[1] / "causal_interventions" / "vllm_backend.py").read_bytes()
            ).hexdigest(),
        },
    }
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    print(canonical_json({"status": "loading_model", "output_dir": str(args.output_dir)}), flush=True)
    started = monotonic()
    if args.backend == "hf":
        backend = LocalHFBackend(snapshot)
    elif args.backend == "mlx":
        backend = LocalMLXBackend(snapshot)
    else:
        backend = LocalVLLMBackend(snapshot, thinking=args.thinking)
    runs = []
    with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for index, episode in enumerate(selected):
            if monotonic() - started >= args.max_wall_seconds:
                break
            run = run_episode(episode, backend, model_id,
                              args.action_max_tokens, args.forecast_max_tokens,
                              score_actions=version,
                              prompt_version="primary-v2" if version else "primary-v1")
            stream.write(canonical_json(run.record()) + "\n")
            stream.flush()
            runs.append(run)
            print(canonical_json({"completed": index + 1, "episode_id": episode.episode_id,
                                  "elapsed_seconds": run.trace["elapsed_seconds"],
                                  "parse_failures": run.trace["parse_failures"]}), flush=True)
    metadata["status"] = "completed" if len(runs) == len(selected) else "wall_time_cap_reached"
    metadata["completed"] = len(runs)
    metadata["total_wall_seconds"] = monotonic() - started
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    if runs:
        references = [r.reference for r in runs]
        results = [result for r in runs for result in r.results]
        if version:
            informative_ids = {e.episode_id for e in selected[:len(runs)]
                               if e.stratum.startswith("informative")}
            report = analyze_v2(references, results, model_id, informative_ids,
                                bootstrap_samples=1000)
            primary = report["primary_action_choice"]
        else:
            report = analyze_paired(references, results, model_id, bootstrap_samples=1000)
            primary = report
        report["pilot_status"] = metadata["status"]
        report["manifest_hash"] = manifest["manifest_hash"]
        (args.output_dir / "summary.json").write_text(canonical_json(report) + "\n")
        print(canonical_json({"status": metadata["status"], "completed": len(runs),
                              "model_minus_random": primary["paired_model_minus_random"],
                              "opportunity_gap": primary["opportunity_gap"],
                              "parse_failures": sum(r.trace["parse_failures"] for r in runs)}), flush=True)
    else:
        print(canonical_json({"status": metadata["status"], "completed": 0}), flush=True)


if __name__ == "__main__":
    main()
