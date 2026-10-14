"""Smoke test the local generation, replay, and oracle audit path."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from causal_interventions.__main__ import _source_hashes


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_freeze_hashes_exclude_appledouble_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "causal_interventions"
            source.mkdir()
            (source / "runner.py").write_text("pass\n")
            (source / "._runner.py").write_bytes(b"AppleDouble metadata")
            rationale = root / "rationale.md"
            rationale.write_text("Fixed count before model test.\n")
            with patch("causal_interventions.__main__.ROOT", root):
                hashes = _source_hashes(rationale)
            self.assertEqual(set(hashes), {"causal_interventions/runner.py", "rationale.md"})

    def test_generate_validate_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            locked_path = Path(directory) / "locked.json"

            def run(*args: str) -> dict:
                completed = subprocess.run(
                    [sys.executable, "-m", "causal_interventions", *args],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return json.loads(completed.stdout)

            created = run(
                "generate", "--config", "configs/benchmark-v0.json", "--output",
                str(path), "--development-per-stratum", "2", "--test-per-stratum", "2",
            )
            checked = run("validate", str(path))
            audited = run("audit", str(path))
            self.assertEqual(created["episodes"], 8)
            self.assertEqual(created["manifest_hash"], checked["manifest_hash"])
            self.assertEqual(created["manifest_hash"], audited["manifest_hash"])
            self.assertEqual(len(audited["groups"]), 4)
            with tempfile.TemporaryDirectory(dir=ROOT) as metadata_dir:
                rationale = Path(metadata_dir) / "count-rationale.txt"
                rationale.write_text("CLI freeze smoke test only; not a study sample count.\n")
                frozen = run(
                    "freeze", "--candidate", str(path), "--output", str(locked_path),
                    "--count-rationale-file", str(rationale),
                )
                verified_lock = run("validate", str(locked_path))
                self.assertEqual(frozen["status"], "locked")
                self.assertEqual(frozen["manifest_hash"], verified_lock["manifest_hash"])
                self.assertEqual(run("audit", str(locked_path))["status"], "locked")


if __name__ == "__main__":
    unittest.main()
