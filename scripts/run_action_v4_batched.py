"""Batch the prespecified two-pass thinking protocol with per-prompt seeds."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import monotonic

from causal_interventions.action_v4 import (action_labels, render_action_prompt,
                                             run_action, summarize_action_runs)
from causal_interventions.benchmark import read_manifest, validate_manifest, verify_freeze_environment
from causal_interventions.episodes import canonical_json
from causal_interventions.runner import Generation
from causal_interventions.vllm_backend import LocalVLLMBackend

ROOT = Path(__file__).resolve().parents[1]


class CachedBackend:
    def __init__(self, generations: dict[str, Generation]) -> None:
        self.generations = generations

    def generate(self, prompt: str, max_new_tokens: int) -> Generation:
        return self.generations[prompt]


def generate_batch(backend: LocalVLLMBackend, prompts: list[str], max_tokens: int,
                   schema: dict) -> tuple[dict[str, Generation], float]:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    if max_tokens <= 256 or not prompts or len(set(prompts)) != len(prompts):
        raise ValueError("nonempty unique prompts and thinking budget above 256 required")
    thought_budget = min(1536, max_tokens - 256)
    rendered = [backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=False, enable_thinking=True) for prompt in prompts]
    seeds = [int.from_bytes(sha256(prompt.encode()).digest()[:4], "big") for prompt in prompts]
    settings = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20,
                               max_tokens=thought_budget, seed=seed) for seed in seeds]
    started = monotonic()
    first = backend.llm.generate(rendered, sampling_params=settings, use_tqdm=False)
    generations: dict[str, Generation] = {}
    pending_prompts: list[str] = []
    pending_prefixes: list[str] = []
    pending_settings = []
    pending_first = []
    for prompt, prefix, seed, response in zip(prompts, rendered, seeds, first, strict=True):
        thought = response.outputs[0]
        if "</think>" in thought.text:
            answer = thought.text.split("</think>", 1)[1].strip()
            try:
                complete = isinstance(json.loads(answer), dict)
            except (TypeError, ValueError):
                complete = False
            if complete:
                generations[prompt] = Generation(
                    answer, len(response.prompt_token_ids), len(thought.token_ids),
                    0.0, raw_text=thought.text)
                continue
            continuation = prefix + thought.text.split("</think>", 1)[0] + "</think>\n"
        else:
            continuation = prefix + thought.text + "\n</think>\n"
        pending_prompts.append(prompt)
        pending_prefixes.append(continuation)
        pending_first.append(response)
        pending_settings.append(SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20,
            max_tokens=max_tokens - len(thought.token_ids), seed=seed + 1,
            guided_decoding=GuidedDecodingParams(json=schema)))
    if pending_prompts:
        second = backend.llm.generate(pending_prefixes, sampling_params=pending_settings,
                                      use_tqdm=False)
        for prompt, previous, response in zip(pending_prompts, pending_first, second, strict=True):
            thought = previous.outputs[0]
            final = response.outputs[0]
            generations[prompt] = Generation(
                final.text, len(previous.prompt_token_ids) + len(response.prompt_token_ids),
                len(thought.token_ids) + len(final.token_ids), 0.0,
                raw_text=thought.text + "\n</think>\n" + final.text)
    elapsed = monotonic() - started
    # The batch wall time is shared. It is not a per-request latency estimate.
    return {prompt: Generation(g.text, g.input_tokens, g.output_tokens, elapsed,
                               raw_text=g.raw_text) for prompt, g in generations.items()}, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--action-max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.max_wall_seconds <= 0 or args.action_max_tokens <= 256
            or args.tensor_parallel_size < 1
            or (args.max_episodes is not None and args.max_episodes < 1)):
        parser.error("invalid limits")
    manifest = read_manifest(args.manifest)
    if manifest["schema_version"] == 2:
        verify_freeze_environment(manifest, ROOT)
        candidate = manifest["candidate"]
    else:
        candidate = manifest
    if candidate["config"]["analysis_version"] != 4:
        parser.error("v4 manifest required")
    if args.split == "test" and manifest["schema_version"] != 2:
        parser.error("test split requires a locked manifest")
    episodes = [episode for episode in validate_manifest(manifest) if episode.split == args.split]
    if args.max_episodes is not None:
        if args.split == "test":
            parser.error("test split must be complete")
        episodes = episodes[:args.max_episodes]
    snapshot = args.model_snapshot.resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        parser.error("model snapshot missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    condition = "opaque_thinking"
    labels = list(action_labels(episodes[0], condition)) + ["STOP"]
    schema = {"type": "object", "additionalProperties": False,
              "required": ["action_id"],
              "properties": {"action_id": {"type": "string", "enum": labels}}}
    metadata = {
        "status": "started", "manifest_hash": manifest["manifest_hash"],
        "snapshot": str(snapshot),
        "model_config_sha256": sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "condition": condition, "split": args.split,
        "backend": "vllm_batched_two_pass", "device": "cuda",
        "python_version": platform.python_version(), "episodes": len(episodes),
        "max_wall_seconds": args.max_wall_seconds,
        "action_max_tokens": args.action_max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "batch_size": args.batch_size,
        "maximum_model_calls": len(episodes),
        "maximum_generated_tokens": len(episodes) * args.action_max_tokens,
        "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "seed": "sha256(prompt)[:32]", "guided_action_schema": True,
                     "thinking": True},
        "episode_ids": [episode.episode_id for episode in episodes],
        "batch_wall_seconds": [],
    }
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    started = monotonic()
    print(canonical_json({"status": "loading_model", "episodes": len(episodes)}), flush=True)
    backend = LocalVLLMBackend(snapshot, thinking=True,
                               tensor_parallel_size=args.tensor_parallel_size,
                               json_schema=schema)
    runs = []
    with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for offset in range(0, len(episodes), args.batch_size):
            if monotonic() - started >= args.max_wall_seconds:
                break
            group = episodes[offset:offset + args.batch_size]
            prompts = [render_action_prompt(episode, condition) for episode in group]
            generations, batch_seconds = generate_batch(backend, prompts,
                                                         args.action_max_tokens, schema)
            metadata["batch_wall_seconds"].append(batch_seconds)
            cached = CachedBackend(generations)
            for episode in group:
                row = run_action(episode, cached, condition, args.action_max_tokens,
                                 allow_locked_test=args.split == "test")
                row["batch_index"] = offset // args.batch_size
                row["batch_size"] = len(group)
                row["batch_wall_seconds"] = batch_seconds
                stream.write(canonical_json(row) + "\n")
                stream.flush()
                runs.append(row)
            metadata["completed"] = len(runs)
            metadata["total_wall_seconds"] = monotonic() - started
            (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
            print(canonical_json({"completed": len(runs), "batch_seconds": batch_seconds,
                                  "invalid_actions": sum(r["invalid_action"] for r in runs)}), flush=True)
    metadata["completed"] = len(runs)
    metadata["total_wall_seconds"] = monotonic() - started
    metadata["status"] = "completed" if len(runs) == len(episodes) else "wall_time_cap_reached"
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    if len(runs) == len(episodes):
        report = summarize_action_runs(episodes, runs)
        report["manifest_hash"] = manifest["manifest_hash"]
        report["condition"] = condition
        report["run_status"] = metadata["status"]
        (args.output_dir / "summary.json").write_text(canonical_json(report) + "\n")
        print(canonical_json({"status": metadata["status"],
                              "model_minus_random": report["model_minus_random"],
                              "interval_95": report["interval_95"]}), flush=True)
    else:
        print(canonical_json({"status": metadata["status"], "completed": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
