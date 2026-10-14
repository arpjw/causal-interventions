"""Run a frozen v5 condition with batched stage calls or bounded calculator calls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import monotonic
from typing import Any

from causal_interventions.action_v5 import (CONDITIONS, STAGE_CONDITIONS, render_stage_prompt,
                                             run_tool_agent, score_stage, stage_schema, summarize,
                                             tool_schema)
from causal_interventions.benchmark import read_manifest, validate_manifest, verify_freeze_environment
from causal_interventions.episodes import Episode, canonical_json
from causal_interventions.runner import Generation
from causal_interventions.vllm_backend import LocalVLLMBackend

ROOT = Path(__file__).resolve().parents[1]


def _seed(prompt: str) -> int:
    return int.from_bytes(sha256(prompt.encode()).digest()[:4], "big")


def generate_stage_batch(backend: LocalVLLMBackend, episodes: list[Episode], condition: str,
                         thought_budget: int) -> tuple[list[tuple[Generation, int, str]], float]:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    if not episodes or condition not in STAGE_CONDITIONS:
        raise ValueError("nonempty valid stage batch required")
    prompts = [render_stage_prompt(episode, condition) for episode in episodes]
    if len(set(prompts)) != len(prompts):
        raise ValueError("stage prompts must be unique")
    schemas = [stage_schema(episode, condition) for episode in episodes]
    seeds = [_seed(prompt) for prompt in prompts]
    rendered = [backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=False, enable_thinking=thought_budget > 0) for prompt in prompts]
    started = monotonic()
    if thought_budget == 0:
        settings = [SamplingParams(
            temperature=0.7, top_p=0.8, top_k=20, max_tokens=512, seed=seed,
            guided_decoding=GuidedDecodingParams(json=schema))
            for seed, schema in zip(seeds, schemas, strict=True)]
        output = backend.llm.generate(rendered, sampling_params=settings, use_tqdm=False)
        elapsed = monotonic() - started
        return [(Generation(item.outputs[0].text, len(item.prompt_token_ids),
                            len(item.outputs[0].token_ids), elapsed), 0,
                 item.outputs[0].finish_reason)
                for item in output], elapsed

    thought_settings = [SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=thought_budget,
        seed=seed) for seed in seeds]
    first = backend.llm.generate(rendered, sampling_params=thought_settings, use_tqdm=False)
    generations: list[tuple[Generation, int, str] | None] = [None] * len(episodes)
    pending: list[tuple[int, str, Any]] = []
    for index, (prefix, response, schema, seed) in enumerate(
            zip(rendered, first, schemas, seeds, strict=True)):
        thought = response.outputs[0]
        answer = thought.text.split("</think>", 1)[1].strip() if "</think>" in thought.text else ""
        try:
            completed = isinstance(json.loads(answer), dict) and set(json.loads(answer)) == set(schema["required"])
        except (TypeError, ValueError):
            completed = False
        if completed:
            generations[index] = (Generation(answer, len(response.prompt_token_ids),
                                             len(thought.token_ids), 0.0, raw_text=thought.text),
                                  len(thought.token_ids), thought.finish_reason)
            continue
        if "</think>" in thought.text:
            continuation = prefix + thought.text.split("</think>", 1)[0] + "</think>\n"
        else:
            continuation = prefix + thought.text + "\n</think>\n"
        pending.append((index, continuation, SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, max_tokens=512, seed=seed + 1,
            guided_decoding=GuidedDecodingParams(json=schema))))
    if pending:
        second = backend.llm.generate([item[1] for item in pending],
                                      sampling_params=[item[2] for item in pending], use_tqdm=False)
        for (index, _, _), response in zip(pending, second, strict=True):
            thought = first[index].outputs[0]
            final = response.outputs[0]
            generations[index] = (Generation(
                final.text, len(first[index].prompt_token_ids) + len(response.prompt_token_ids),
                len(thought.token_ids) + len(final.token_ids), 0.0,
                raw_text=thought.text + "\n</think>\n" + final.text),
                len(thought.token_ids), thought.finish_reason)
    elapsed = monotonic() - started
    if any(item is None for item in generations):
        raise AssertionError("missing batched generation")
    return [(Generation(g.text, g.input_tokens, g.output_tokens, elapsed, raw_text=g.raw_text),
             thought_tokens, finish_reason)
            for g, thought_tokens, finish_reason in generations if g is not None], elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "test"), required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if (args.max_wall_seconds <= 0 or args.batch_size < 1 or args.tensor_parallel_size < 1
            or (args.max_episodes is not None and args.max_episodes < 1)):
        parser.error("invalid limits")
    if args.split == "test" and args.max_episodes is not None:
        parser.error("test must include every episode")
    manifest = read_manifest(args.manifest)
    if manifest["schema_version"] != 2:
        parser.error("locked manifest required")
    verify_freeze_environment(manifest, ROOT)
    if manifest["candidate"]["config"]["seed"] != 827419:
        parser.error("v5 seed required")
    episodes = [episode for episode in validate_manifest(manifest) if episode.split == args.split]
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    snapshot = args.model_snapshot.resolve()
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        parser.error("model snapshot missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    thinking = args.condition.startswith("q235_")
    thought_budget = (3584 if args.condition == "q235_long_raw_stages" else 1536) if thinking else 0
    metadata = {
        "status": "started", "manifest_hash": manifest["manifest_hash"],
        "condition": args.condition, "split": args.split, "episodes": len(episodes),
        "snapshot": str(snapshot), "model_config_sha256": sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "device": "cuda", "backend": "vllm_0.10.2",
        "max_wall_seconds": args.max_wall_seconds, "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size, "thought_budget": thought_budget,
        "final_token_budget": 512,
        "decoding": {"temperature": 0.6 if thinking else 0.7,
                     "top_p": 0.95 if thinking else 0.8, "top_k": 20,
                     "seed": "sha256(prompt)[:32]", "guided_json_schema": True},
        "episode_ids": [episode.episode_id for episode in episodes],
        "batch_wall_seconds": [],
    }
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    started = monotonic()
    print(canonical_json({"status": "loading_model", "episodes": len(episodes)}), flush=True)
    backend = LocalVLLMBackend(snapshot, thinking=thinking,
                               tensor_parallel_size=args.tensor_parallel_size,
                               json_schema=tool_schema() if args.condition == "q32_calculator_agent" else None)
    rows: list[dict[str, Any]] = []
    with (args.output_dir / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for offset in range(0, len(episodes), args.batch_size):
            if monotonic() - started >= args.max_wall_seconds:
                break
            group = episodes[offset:offset + args.batch_size]
            if args.condition == "q32_calculator_agent":
                current = [run_tool_agent(episode, backend, allow_locked_test=args.split == "test")
                           for episode in group]
                elapsed = sum(sum(call["generation"]["elapsed_seconds"] for call in row["calls"])
                              for row in current)
            else:
                generations, elapsed = generate_stage_batch(backend, group, args.condition,
                                                             thought_budget)
                current = []
                for episode, (generation, thought_tokens, finish_reason) in zip(
                        group, generations, strict=True):
                    row = score_stage(episode, args.condition, generation,
                                      allow_locked_test=args.split == "test")
                    row["thought_tokens"] = thought_tokens
                    row["thought_finish_reason"] = finish_reason
                    current.append(row)
            rows.extend(current)
            for row in current:
                stream.write(canonical_json(row) + "\n")
                stream.flush()
            metadata["completed"] = len(rows)
            metadata["total_wall_seconds"] = monotonic() - started
            metadata["batch_wall_seconds"].append(elapsed)
            (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
            print(canonical_json({"completed": len(rows), "batch_seconds": elapsed,
                                  "invalid_actions": sum(row["invalid_action"] for row in rows)}), flush=True)
    metadata["completed"] = len(rows)
    metadata["total_wall_seconds"] = monotonic() - started
    metadata["status"] = "completed" if len(rows) == len(episodes) else "wall_time_cap_reached"
    (args.output_dir / "metadata.json").write_text(canonical_json(metadata) + "\n")
    if len(rows) == len(episodes):
        report = summarize(episodes, rows)
        report.update({"manifest_hash": manifest["manifest_hash"], "condition": args.condition,
                       "run_status": metadata["status"]})
        (args.output_dir / "summary.json").write_text(canonical_json(report) + "\n")
        print(canonical_json({"status": "completed", "model_minus_random": report["model_minus_random"],
                              "interval_95": report["interval_95"]}), flush=True)
    else:
        print(canonical_json({"status": metadata["status"], "completed": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
