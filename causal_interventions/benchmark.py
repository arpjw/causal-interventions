"""Serializable split manifests, integrity checks, and immutable local writes."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from typing import Any

from .episodes import (Episode, GeneratorConfig, VARIABLES, canonical_json, digest,
                       generate_split, joint_from_parameters, one_step_diagnostics,
                       rejection_reason)
from .types import Intervention, Query
from .worlds import FiniteSCM


def analysis_specification(version: int = 1) -> dict[str, Any]:
    """Return a fresh canonical analysis contract for this unfrozen schema."""
    specification = {
        "primary_metric": "mean_excess_log_loss_nats",
        "uncertainty_unit": "independent_joint_parameter_draw",
        "primary_comparison": "paired_model_minus_random",
        "minimum_gap_for_gap_closure": 0.000001,
        "stratify_by": ["stratum", "candidate_count", "budget"],
        "freeze_requires": [
            "exact_policy_opportunity_audit", "source_hashes", "prompt_hash",
            "analysis_script_hash", "pilot_count_rationale",
        ],
    }
    if version == 2:
        specification.update(
            primary_metric="exact_bayesian_forecast_after_model_actions_excess_log_loss_nats",
            primary_comparison="model_action_bayes_minus_random_on_informative_strata",
            secondary_metrics=["submitted_forecast_excess_log_loss_nats", "near_tie_action_loss"],
        )
    elif version == 3:
        specification.update(
            primary_metric="prior_averaged_one_step_bayes_risk_nats",
            primary_comparison="chosen_action_risk_minus_uniform_random_on_informative_strata",
            secondary_metrics=["oracle_gap_closure", "near_tie_action_risk", "invalid_action_rate"],
        )
    elif version == 4:
        specification.update(
            primary_metric="prior_averaged_one_step_bayes_risk_nats",
            primary_comparison="chosen_action_risk_minus_uniform_random_on_informative_strata_v4",
            secondary_metrics=["opaque_label_effect", "likelihood_table_effect",
                               "action_value_effect", "large_thinking_model_effect",
                               "near_tie_action_risk", "invalid_action_rate"],
        )
    elif version != 1:
        raise ValueError("unsupported analysis version")
    return specification


def _portable_equal(expected: Any, observed: Any) -> bool:
    """Match deterministic replay across libm/Python versions without weakening IDs.

    Manifest and per-record hashes still bind the serialized bytes exactly. This
    tolerance applies only to newly recomputed floating point calculations.
    """
    if isinstance(expected, float) and isinstance(observed, (float, int)):
        return math.isclose(expected, observed, rel_tol=0, abs_tol=1e-12)
    if type(expected) is not type(observed):
        return False
    if isinstance(expected, dict):
        return expected.keys() == observed.keys() and all(
            _portable_equal(expected[key], observed[key]) for key in expected)
    if isinstance(expected, (tuple, list)):
        return len(expected) == len(observed) and all(
            _portable_equal(a, b) for a, b in zip(expected, observed, strict=True))
    return expected == observed


def episode_from_record(record: dict[str, Any]) -> Episode:
    body = {k: v for k, v in record.items() if k != "validation_hash"}
    if record.get("validation_hash") != digest(body) or record.get("schema_version") != 1:
        raise ValueError("episode hash or schema mismatch")
    if tuple(record["variables"]) != VARIABLES:
        raise ValueError("unsupported episode variables")
    parameters = tuple(record["parameters"])
    if digest(parameters) != record["parameter_id"]:
        raise ValueError("parameter hash mismatch")
    joint = tuple(record["joint"])
    if not _portable_equal(joint, joint_from_parameters(parameters)):
        raise ValueError("joint differs from declared parameters")
    worlds = tuple(FiniteSCM(VARIABLES, tuple(order), joint) for order in record["orders"])
    if len({w.order for w in worlds}) != len(worlds) or not 2 <= len(worlds) <= 6:
        raise ValueError("candidate orders must be distinct")
    actions = tuple(Intervention(**a) for a in record["actions"])
    queries = tuple(Query(Intervention(**q["intervention"]), q["outcome"], q["outcome_value"])
                    for q in record["queries"])
    if not queries or len(actions) < 2 or len(set(actions)) != len(actions):
        raise ValueError("invalid actions or queries")
    if any(a == q.intervention or a.target == q.outcome for a in actions for q in queries):
        raise ValueError("query intervention or outcome leaked into exploration")
    prior = tuple(record["prior"])
    if prior != tuple(1 / len(worlds) for _ in worlds):
        raise ValueError("primary benchmark prior must be uniform")
    if record["budget"] not in (1, 2, 3) or not 0 <= record["hidden_world"] < len(worlds):
        raise ValueError("invalid budget or hidden world")
    diagnostics = one_step_diagnostics(worlds, prior, actions, queries)
    if not _portable_equal(diagnostics, record["diagnostics"]):
        raise ValueError("diagnostics mismatch")
    expected_id = digest([record["split"], record["stratum"], record["parameter_id"],
                          record["orders"], record["budget"]])
    if expected_id != record["episode_id"]:
        raise ValueError("episode ID mismatch")
    return Episode(record["episode_id"], record["parameter_id"], record["split"],
                   record["stratum"], record["generator_seed"], parameters, worlds, prior,
                   actions, queries, record["budget"], record["hidden_world"],
                   record["outcome_seed"], diagnostics)


def build_manifest(config: GeneratorConfig, development_per_stratum: int = 4,
                   test_per_stratum: int = 4) -> dict[str, Any]:
    development, dev_counts = generate_split(config, "development", development_per_stratum)
    test, test_counts = generate_split(config, "test", test_per_stratum)
    manifest: dict[str, Any] = {
        "schema_version": 1, "status": "candidate_unfrozen", "config": json.loads(canonical_json(asdict(config))),
        "split_counts": {"development": dev_counts, "test": test_counts},
        "episodes": [e.record() for e in development + test],
        "analysis": analysis_specification(config.analysis_version),
    }
    if config.analysis_version in (3, 4):
        manifest["generator_python"] = platform.python_version()
    manifest["manifest_hash"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> tuple[Episode, ...]:
    if manifest.get("schema_version") == 2:
        return _validate_locked_manifest(manifest)
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if manifest.get("schema_version") != 1 or manifest.get("manifest_hash") != digest(body):
        raise ValueError("manifest hash or schema mismatch")
    if manifest.get("status") != "candidate_unfrozen":
        raise ValueError("schema 1 only supports candidate_unfrozen status")
    if canonical_json(manifest.get("analysis")) != canonical_json(
        analysis_specification(manifest.get("config", {}).get("analysis_version", 1))
    ):
        raise ValueError("analysis specification mismatch")
    episodes = tuple(episode_from_record(record) for record in manifest["episodes"])
    if not episodes or len({e.episode_id for e in episodes}) != len(episodes):
        raise ValueError("empty manifest or repeated episode")
    if len({e.parameter_id for e in episodes}) != len(episodes):
        raise ValueError("parameter overlap within or across splits")
    from .episodes import Stratum
    settings = dict(manifest["config"])
    settings["strata"] = tuple(Stratum(**s) for s in settings["strata"])
    config = GeneratorConfig(**settings)
    strata = {s.name: s for s in config.strata}
    for episode in episodes:
        if episode.split not in ("development", "test") or episode.stratum not in strata:
            raise ValueError("unknown split or stratum")
        stratum = strata[episode.stratum]
        if episode.budget != stratum.budget or len(episode.worlds) != stratum.candidate_count:
            raise ValueError("episode does not match stratum")
        if rejection_reason(episode.worlds[0].joint, episode.diagnostics, config, stratum):
            raise ValueError("episode fails declared rejection rules")
    for split in ("development", "test"):
        for stratum in strata:
            counts = manifest["split_counts"][split][stratum]
            if counts["accepted"] != sum(e.split == split and e.stratum == stratum for e in episodes):
                raise ValueError("accepted count mismatch")
            if counts["attempted"] != counts["accepted"] + sum(counts["rejected"].values()):
                raise ValueError("rejection accounting mismatch")
    # Reproduce the complete generator output on the generating Python runtime.
    # Cross-version libm differences can change a rejection at a strict gate,
    # shifting all later RNG draws. In that case exact hashes and per-record
    # re-factorization still hold, but the full generator replay is local-only.
    generating_python = manifest.get("generator_python")
    if generating_python is not None and (
        config.analysis_version not in (3, 4) or not isinstance(generating_python, str)
        or not generating_python
    ):
        raise ValueError("invalid generator runtime provenance")
    if generating_python is not None and generating_python != platform.python_version():
        return episodes
    expected_records = []
    for split in ("development", "test"):
        split_counts = manifest["split_counts"][split]
        accepted_counts = {split_counts[name]["accepted"] for name in strata}
        if len(accepted_counts) != 1:
            raise ValueError("strata must have balanced accepted counts")
        count = accepted_counts.pop()
        regenerated, regenerated_counts = generate_split(config, split, count)
        if regenerated_counts != split_counts:
            raise ValueError("rejection counts do not reproduce from config")
        expected_records.extend(episode.record() for episode in regenerated)
    if not _portable_equal(
        [{k: v for k, v in record.items() if k != "validation_hash"} for record in expected_records],
        [{k: v for k, v in record.items() if k != "validation_hash"} for record in manifest["episodes"]],
    ):
        raise ValueError("episode records do not reproduce from config")
    return episodes


def baseline_audit(candidate: dict[str, Any]) -> dict[str, Any]:
    """Integrate all policy outcomes at each episode's full declared budget."""
    if candidate.get("schema_version") != 1:
        raise ValueError("baseline audit requires a candidate manifest")
    episodes = validate_manifest(candidate)
    return _baseline_audit(candidate, episodes)


def _baseline_audit(candidate: dict[str, Any], episodes: tuple[Episode, ...]) -> dict[str, Any]:
    from .beliefs import BeliefModel
    from .policies import ExactPolicy, RandomPolicy, expected_policy_risk

    groups: dict[tuple[str, str], list[float]] = {}
    for episode in episodes:
        model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
        gap = expected_policy_risk(model, RandomPolicy(model), model.prior, episode.budget)
        gap -= ExactPolicy(model).value(model.prior, episode.budget)
        groups.setdefault((episode.split, episode.stratum), []).append(gap)
    return {
        "manifest_hash": candidate["manifest_hash"],
        "status": candidate["status"],
        "groups": [
            {"split": split, "stratum": stratum, "episodes": len(values),
             "min_opportunity_gap_nats": min(values),
             "mean_opportunity_gap_nats": sum(values) / len(values),
             "max_opportunity_gap_nats": max(values)}
            for (split, stratum), values in sorted(groups.items())
        ],
    }


def _valid_hash(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _relative_file(path: Any) -> bool:
    return (isinstance(path, str) and bool(path) and not Path(path).is_absolute()
            and ".." not in Path(path).parts and Path(path).as_posix() == path
            and path != ".")


def freeze_manifest(candidate: dict[str, Any], source_hashes: dict[str, str],
                    prompt_hash: str, analysis_script_hash: str, count_rationale: str,
                    baseline_audit: dict[str, Any]) -> dict[str, Any]:
    """Lock a candidate with evidence; this operation does not write files.

    Prompt and analysis hashes must identify files in source_hashes. All paths
    are relative to the repository root. Verify their current bytes separately
    with verify_freeze_environment before running an evaluation.
    """
    if candidate.get("schema_version") != 1:
        raise ValueError("only a schema 1 candidate can be frozen")
    manifest = {
        "schema_version": 2, "status": "locked", "candidate": candidate,
        "freeze": {"source_hashes": source_hashes, "prompt_hash": prompt_hash,
                   "analysis_script_hash": analysis_script_hash,
                   "count_rationale": count_rationale, "baseline_audit": baseline_audit},
    }
    # Own all nested data so later mutation of caller arguments cannot alter a lock.
    manifest = json.loads(canonical_json(manifest))
    manifest["manifest_hash"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def _validate_locked_manifest(manifest: dict[str, Any]) -> tuple[Episode, ...]:
    expected_keys = {"schema_version", "status", "candidate", "freeze", "manifest_hash"}
    body = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    if (set(manifest) != expected_keys or manifest.get("status") != "locked"
            or manifest.get("manifest_hash") != digest(body)):
        raise ValueError("locked manifest hash, status, or schema mismatch")
    candidate = manifest["candidate"]
    if not isinstance(candidate, dict) or candidate.get("schema_version") != 1:
        raise ValueError("locked manifest must contain a schema 1 candidate")
    episodes = validate_manifest(candidate)
    freeze = manifest["freeze"]
    freeze_keys = {"source_hashes", "prompt_hash", "analysis_script_hash",
                   "count_rationale", "baseline_audit"}
    if not isinstance(freeze, dict) or set(freeze) != freeze_keys:
        raise ValueError("invalid freeze evidence schema")
    hashes = freeze["source_hashes"]
    if (not isinstance(hashes, dict) or not hashes
            or any(not _relative_file(path) or not _valid_hash(value)
                   for path, value in hashes.items())):
        raise ValueError("source hashes must name relative files with SHA256 hashes")
    for field in ("prompt_hash", "analysis_script_hash"):
        if not _valid_hash(freeze[field]) or freeze[field] not in hashes.values():
            raise ValueError(f"{field} must identify a hashed source file")
    rationale = freeze["count_rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("nonempty sample-count rationale is required")
    audit = _baseline_audit(candidate, episodes)
    if not _portable_equal(freeze["baseline_audit"], audit):
        raise ValueError("full-horizon baseline audit does not reproduce")
    bounds = {s["name"]: s["min_opportunity_gap"] for s in candidate["config"]["strata"]}
    if not any(bound > 0 for bound in bounds.values()):
        raise ValueError("freeze requires an informative stratum with a positive gap threshold")
    for group in audit["groups"]:
        bound = bounds[group["stratum"]]
        if bound > 0 and group["min_opportunity_gap_nats"] < bound:
            raise ValueError("full-horizon opportunity gap fails the informative stratum threshold")
    return episodes


def verify_freeze_environment(manifest: dict[str, Any], repository_root: str | Path) -> None:
    """Validate the lock and every source, prompt, and analysis file's bytes."""
    if manifest.get("schema_version") != 2:
        raise ValueError("environment verification requires a locked manifest")
    validate_manifest(manifest)
    root = Path(repository_root).resolve()
    for relative, expected in manifest["freeze"]["source_hashes"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"frozen file missing or outside repository: {relative}")
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen file hash mismatch: {relative}")


def write_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(manifest) + "\n")


def read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        result = json.load(stream)
    validate_manifest(result)
    return result
