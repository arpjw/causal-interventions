"""Local, model-free benchmark generation and validation commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from .benchmark import (
    baseline_audit, build_manifest, freeze_manifest, read_manifest,
    verify_freeze_environment, write_manifest,
)
from .episodes import GeneratorConfig, Stratum, canonical_json, digest

ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path | None) -> GeneratorConfig:
    if path is None:
        return GeneratorConfig()
    with path.open(encoding="utf-8") as stream:
        settings = json.load(stream)
    settings["strata"] = tuple(Stratum(**item) for item in settings["strata"])
    return GeneratorConfig(**settings)


def _audit(path: Path) -> dict[str, object]:
    manifest = read_manifest(path)
    candidate = manifest["candidate"] if manifest["schema_version"] == 2 else manifest
    result = baseline_audit(candidate)
    if manifest["schema_version"] == 2:
        result = {**result, "status": "locked", "lock_hash": manifest["manifest_hash"]}
    return result


def _source_hashes(rationale_path: Path) -> dict[str, str]:
    # macOS transfers can create AppleDouble files such as ._runner.py.
    # They are filesystem metadata, not study source.
    paths = [path for pattern in (
        "causal_interventions/*.py", "scripts/*.py", "configs/*.json",
        "prompts/*.txt", "requirements-*.txt"
    ) for path in sorted(ROOT.glob(pattern)) if not path.name.startswith("._")]
    paths.append(rationale_path.resolve())
    hashes = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT) or not resolved.is_file():
            raise ValueError(f"freeze source is outside repository or missing: {path}")
        hashes[resolved.relative_to(ROOT).as_posix()] = sha256(resolved.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="write an unfrozen candidate manifest")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--config", type=Path)
    generate.add_argument("--development-per-stratum", type=int, default=4)
    generate.add_argument("--test-per-stratum", type=int, default=4)
    validate = commands.add_parser("validate", help="replay and validate a manifest")
    validate.add_argument("path", type=Path)
    audit = commands.add_parser("audit", help="compare exact and random policy risks")
    audit.add_argument("path", type=Path)
    freeze = commands.add_parser("freeze", help="lock a candidate with local source evidence")
    freeze.add_argument("--candidate", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--count-rationale-file", type=Path, required=True)
    freeze.add_argument("--prompt", default="prompts/primary-v1.txt")
    freeze.add_argument("--analysis-script", default="causal_interventions/analysis.py")
    args = parser.parse_args()

    if args.command == "generate":
        config = _load_config(args.config)
        manifest = build_manifest(
            config, args.development_per_stratum, args.test_per_stratum
        )
        write_manifest(manifest, args.output)
        print(
            canonical_json(
                {
                    "path": str(args.output),
                    "manifest_hash": manifest["manifest_hash"],
                    "status": manifest["status"],
                    "config_hash": digest(asdict(config)),
                    "strata": len(config.strata),
                    "episodes": len(manifest["episodes"]),
                }
            )
        )
    elif args.command == "validate":
        manifest = read_manifest(args.path)
        if manifest["schema_version"] == 2:
            verify_freeze_environment(manifest, ROOT)
        candidate = manifest["candidate"] if manifest["schema_version"] == 2 else manifest
        print(
            canonical_json(
                {
                    "manifest_hash": manifest["manifest_hash"],
                    "status": manifest["status"],
                    "episodes": len(candidate["episodes"]),
                    "valid": True,
                }
            )
        )
    elif args.command == "audit":
        print(canonical_json(_audit(args.path)))
    else:
        candidate = read_manifest(args.candidate)
        rationale_path = args.count_rationale_file.resolve()
        source_hashes = _source_hashes(rationale_path)
        prompt_hash = source_hashes[args.prompt]
        analysis_hash = source_hashes[args.analysis_script]
        locked = freeze_manifest(
            candidate, source_hashes, prompt_hash, analysis_hash,
            rationale_path.read_text(encoding="utf-8"), baseline_audit(candidate),
        )
        verify_freeze_environment(locked, ROOT)
        write_manifest(locked, args.output)
        print(canonical_json({"path": str(args.output), "manifest_hash": locked["manifest_hash"],
                              "status": "locked", "episodes": len(candidate["episodes"])}))


if __name__ == "__main__":
    main()
