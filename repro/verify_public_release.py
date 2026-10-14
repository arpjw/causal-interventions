"""Verify published file hashes and optionally replay every v2-v5 test ledger.

This uses only the Python standard library. It does not download model weights,
call a model, alter the frozen test sets, or write into the repository.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
MANIFESTS = RELEASE / "manifests"
LEDGERS = RELEASE / "ledgers"


def check_hashes() -> None:
    index = json.loads((RELEASE / "sha256.json").read_text())
    if not isinstance(index, dict) or not index:
        raise ValueError("empty or malformed release index")
    total = 0
    for relative, expected in index.items():
        path = (RELEASE / relative).resolve()
        if not path.is_relative_to(RELEASE.resolve()) or not path.is_file():
            raise ValueError(f"missing or unsafe indexed path: {relative}")
        data = path.read_bytes()
        if len(data) != expected["bytes"] or sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"size or SHA-256 mismatch: {relative}")
        total += len(data)
    print(f"Verified {len(index)} indexed evidence files ({total:,} bytes)", flush=True)


def run(script: Path, manifest: Path, ledger: Path | None = None, *, root: Path = ROOT) -> None:
    command = [sys.executable, str(script), str(manifest)]
    if ledger is not None:
        command.append(str(ledger))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    subprocess.run(command, cwd=root, env=environment, check=True)


def extract_frozen_v2(destination: Path) -> None:
    archive_path = RELEASE / "frozen-source" / "action-v2-source.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError(f"unexpected archive member: {member.name}")
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"unsafe archive path: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    verifier = destination / "scripts" / "verify_action_v2.py"
    verifier.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts" / "verify_action_v2.py", verifier)


def replay() -> None:
    with TemporaryDirectory(prefix="causal-v2-replay-") as directory:
        frozen = Path(directory)
        extract_frozen_v2(frozen)
        for name in ("runpod-action-v2-locked32", "runpod-action-v2-locked72"):
            run(frozen / "scripts" / "verify_action_v2.py",
                MANIFESTS / "action-v2-locked.json", LEDGERS / name, root=frozen)

    for name in ("v3-test32-base", "v3-test32-assisted", "v3-test72-base",
                 "v3-test72-assisted", "v3-test235-base"):
        run(ROOT / "scripts" / "verify_action_v3_portable.py",
            MANIFESTS / "action-v3-final-locked.json", LEDGERS / name)
    for name in ("v4-test32-literal-raw", "v4-test32-opaque-raw",
                 "v4-test32-opaque-likelihood", "v4-test32-opaque-values",
                 "v4-test235-opaque-thinking-batched"):
        run(ROOT / "scripts" / "verify_action_v4.py",
            MANIFESTS / "action-v4-locked.json", LEDGERS / name)
    for name in ("v5-test32-raw", "v5-test32-tool"):
        run(ROOT / "scripts" / "verify_action_v5.py",
            MANIFESTS / "action-v5-locked.json", LEDGERS / name)
    for name in ("action-v3-final-locked.json", "action-v4-locked.json",
                 "action-v5-locked.json"):
        run(ROOT / "repro" / "check_information_identity.py", MANIFESTS / name)
    print("Replayed 14 complete v2-v5 test ledgers and checked the v3-v5 information identity", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="also replay all complete v2-v5 test ledgers and the information identity")
    args = parser.parse_args()
    check_hashes()
    if args.full:
        replay()


if __name__ == "__main__":
    main()
