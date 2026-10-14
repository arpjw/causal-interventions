"""Generator gates, split isolation, and serialization integrity."""
import copy
from hashlib import sha256
import json
import tempfile
import unittest
from pathlib import Path

from causal_interventions.benchmark import (baseline_audit, build_manifest, freeze_manifest,
                                           read_manifest, validate_manifest,
                                           verify_freeze_environment, write_manifest)
from causal_interventions.episodes import GeneratorConfig, Stratum, STATES, digest, generate_split


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = GeneratorConfig()
        cls.manifest = build_manifest(cls.config, 2, 2)
        cls.episodes = validate_manifest(cls.manifest)

    def test_deterministic_and_parameter_disjoint(self):
        self.assertEqual(self.manifest, build_manifest(self.config, 2, 2))
        dev = {e.parameter_id for e in self.episodes if e.split == "development"}
        test = {e.parameter_id for e in self.episodes if e.split == "test"}
        self.assertFalse(dev & test)
        self.assertEqual(len(dev), 4)
        self.assertEqual(len(test), 4)

    def test_equivalence_query_safety_and_opportunity(self):
        for episode in self.episodes:
            self.assertGreaterEqual(min(episode.worlds[0].joint), self.config.min_cell)
            for world in episode.worlds:
                for got, expected in zip((world.observational_probability(state) for state in STATES), episode.worlds[0].joint):
                    self.assertAlmostEqual(got, expected, places=12)
            for query in episode.queries:
                self.assertNotIn(query.intervention, episode.actions)
                self.assertTrue(all(a.target != query.outcome for a in episode.actions))
            self.assertNotIn("episode_id", episode.public_record())
            self.assertNotIn("hidden_world", episode.public_record())
            self.assertNotIn("outcome_seed", episode.public_record())
            if episode.stratum == "informative":
                self.assertGreaterEqual(episode.diagnostics["action_gap"], 0.00001)
                self.assertGreaterEqual(episode.diagnostics["one_step_opportunity_gap"], 0.001)
                self.assertGreaterEqual(episode.diagnostics["second_action_gain"], 0.000001)
            else:
                self.assertLessEqual(episode.diagnostics["action_gap"], 0.00001)

    def test_independent_diagnostics_match_decision_core(self):
        from causal_interventions.beliefs import BeliefModel
        from causal_interventions.policies import ExactPolicy, RandomPolicy, expected_policy_risk
        for episode in self.episodes:
            model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
            exact = ExactPolicy(model)
            self.assertAlmostEqual(model.risk(model.prior), episode.diagnostics["initial_risk"], places=12)
            self.assertAlmostEqual(exact.value(model.prior, 1),
                                   episode.diagnostics["one_step_oracle_risk"], places=12)
            self.assertAlmostEqual(expected_policy_risk(model, RandomPolicy(model), model.prior, 1),
                                   episode.diagnostics["one_step_random_risk"], places=12)

    def test_roundtrip_and_no_overwrite(self):
        self.assertEqual(validate_manifest(json.loads(json.dumps(self.manifest))), self.episodes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(self.manifest, path)
            self.assertEqual(read_manifest(path), self.manifest)
            with self.assertRaises(FileExistsError):
                write_manifest(self.manifest, path)

    def test_tamper_rejected_even_if_rehashed(self):
        corrupted = copy.deepcopy(self.manifest)
        corrupted["episodes"][0]["joint"][0] += 0.01
        with self.assertRaises(ValueError):
            validate_manifest(corrupted)
        record = corrupted["episodes"][0]
        record["validation_hash"] = digest({k: v for k, v in record.items() if k != "validation_hash"})
        corrupted["manifest_hash"] = digest({k: v for k, v in corrupted.items() if k != "manifest_hash"})
        with self.assertRaises(ValueError):
            validate_manifest(corrupted)

    def test_rehashed_private_state_tamper_fails_replay(self):
        for field in ("hidden_world", "outcome_seed", "generator_seed"):
            corrupted = copy.deepcopy(self.manifest)
            record = corrupted["episodes"][0]
            record[field] += 1
            if field == "hidden_world":
                record[field] %= len(record["orders"])
            record["validation_hash"] = digest({k: v for k, v in record.items() if k != "validation_hash"})
            corrupted["manifest_hash"] = digest({k: v for k, v in corrupted.items() if k != "manifest_hash"})
            with self.assertRaisesRegex(ValueError, "do not reproduce"):
                validate_manifest(corrupted)

    def test_rehashed_status_or_analysis_tamper_rejected(self):
        changes = (
            ("status", "locked", "candidate_unfrozen"),
            ("analysis", {"primary_metric": "made_up"}, "analysis specification"),
        )
        for field, value, expected_error in changes:
            with self.subTest(field=field):
                corrupted = copy.deepcopy(self.manifest)
                corrupted[field] = value
                corrupted["manifest_hash"] = digest({k: v for k, v in corrupted.items()
                                                     if k != "manifest_hash"})
                with self.assertRaisesRegex(ValueError, expected_error):
                    validate_manifest(corrupted)
        corrupted = copy.deepcopy(self.manifest)
        corrupted["analysis"]["freeze_requires"].remove("prompt_hash")
        corrupted["manifest_hash"] = digest({k: v for k, v in corrupted.items()
                                             if k != "manifest_hash"})
        with self.assertRaisesRegex(ValueError, "analysis specification"):
            validate_manifest(corrupted)

    def test_freeze_roundtrip_environment_and_detached_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"core.py": "# source\n", "prompt.txt": "forecast query\n",
                     "analysis.py": "# analysis\n"}
            hashes = {}
            for name, contents in files.items():
                (root / name).write_text(contents)
                hashes[name] = sha256(contents.encode()).hexdigest()
            candidate = copy.deepcopy(self.manifest)
            locked = freeze_manifest(candidate, hashes, hashes["prompt.txt"],
                                     hashes["analysis.py"], "Pilot supports these fixed counts.",
                                     baseline_audit(candidate))
            self.assertEqual(locked["status"], "locked")
            self.assertEqual(validate_manifest(locked), self.episodes)
            verify_freeze_environment(locked, root)
            candidate["status"] = "changed"
            hashes.clear()
            self.assertEqual(locked["candidate"]["status"], "candidate_unfrozen")
            path = root / "locked.json"
            write_manifest(locked, path)
            self.assertEqual(read_manifest(path), locked)
            (root / "prompt.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_freeze_environment(locked, root)

    def test_freeze_rejects_missing_or_rehashed_evidence(self):
        hashes = {"core.py": "a" * 64, "prompt.txt": "b" * 64, "analysis.py": "c" * 64}
        locked = freeze_manifest(self.manifest, hashes, hashes["prompt.txt"],
                                 hashes["analysis.py"], "Fixed pilot count.",
                                 baseline_audit(self.manifest))
        changes = (
            ("count_rationale", " "), ("prompt_hash", ""),
            ("analysis_script_hash", "d" * 64),
            ("source_hashes", {"../outside.py": "a" * 64}),
            ("baseline_audit", {}),
        )
        for field, value in changes:
            with self.subTest(field=field):
                corrupted = copy.deepcopy(locked)
                corrupted["freeze"][field] = value
                corrupted["manifest_hash"] = digest({k: v for k, v in corrupted.items()
                                                     if k != "manifest_hash"})
                with self.assertRaises(ValueError):
                    validate_manifest(corrupted)

    def test_freeze_audit_uses_full_budget(self):
        candidate = build_manifest(GeneratorConfig(strata=(Stratum("informative", budget=2),)), 1, 1)
        audit = baseline_audit(candidate)
        episodes = validate_manifest(candidate)
        from causal_interventions.beliefs import BeliefModel
        from causal_interventions.policies import ExactPolicy, RandomPolicy, expected_policy_risk
        for episode, group in zip(episodes, audit["groups"]):
            model = BeliefModel(episode.worlds, episode.actions, episode.queries, episode.prior)
            expected = expected_policy_risk(model, RandomPolicy(model), model.prior, 2)
            expected -= ExactPolicy(model).value(model.prior, 2)
            self.assertEqual(group["mean_opportunity_gap_nats"], expected)
            self.assertNotAlmostEqual(expected, episode.diagnostics["one_step_opportunity_gap"], places=8)

    def test_rejections_are_counted_and_impossible_gate_fails(self):
        for counts in self.manifest["split_counts"].values():
            for row in counts.values():
                self.assertEqual(row["attempted"], row["accepted"] + sum(row["rejected"].values()))
        config = GeneratorConfig(max_attempts_per_stratum=3,
                                 strata=(Stratum("impossible", min_action_gap=0.9),))
        with self.assertRaisesRegex(ValueError, "exhausted attempt limit"):
            generate_split(config, "development", 1)


if __name__ == "__main__":
    unittest.main()
