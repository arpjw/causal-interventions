"""Copy a fixed allowlist of locked manifests and ledgers into release/."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
MANIFESTS = (
    "qwen32-locked-v1.json",
    "action-v2-locked.json",
    "action-v3-final-locked.json",
    "action-v4-locked.json",
    "action-v5-locked.json",
)
LEDGERS = (
    "runpod-qwen32-awq-locked-v1",
    "runpod-action-v2-locked32",
    "runpod-action-v2-locked72",
    "v3-test32-base",
    "v3-test32-assisted",
    "v3-test72-base",
    "v3-test72-assisted",
    "v3-test235-base",
    "v4-test32-literal-raw",
    "v4-test32-opaque-raw",
    "v4-test32-opaque-likelihood",
    "v4-test32-opaque-values",
    "v4-test235-opaque-thinking-batched",
    "v5-dev32-raw",
    "v5-dev32-raw-512",
    "v5-dev32-tool",
    "v5-dev235-short8",
    "v5-test32-raw",
    "v5-test32-tool",
)
VERSIONS = ("v2", "v3", "v4", "v5")


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    for name in MANIFESTS:
        copy(ROOT / "data" / name, RELEASE / "manifests" / name)
    for name in LEDGERS:
        source = ROOT / "results" / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        for filename in ("metadata.json", "runs.jsonl", "summary.json"):
            copy(source / filename, RELEASE / "ledgers" / name / filename)
        if name == "runpod-qwen32-awq-locked-v1":
            copy(source / "provider.json", RELEASE / "ledgers" / name / "provider.json")
    frozen = RELEASE / "frozen-source"
    frozen.mkdir(parents=True, exist_ok=True)
    copy(ROOT / "results" / "runpod-qwen32-locked-sources-v1.tar.gz",
         frozen / "qwen32-locked-sources-v1.tar.gz")
    for version in VERSIONS:
        source = ROOT / "artifacts" / f"action-{version}-source"
        if not source.is_dir():
            raise FileNotFoundError(source)
        with tarfile.open(frozen / f"action-{version}-source.tar.gz", "w:gz") as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith("._"):
                    archive.add(path, arcname=path.relative_to(source))
    copy(ROOT / "artifacts" / "action-v4-source-amendment" / "scripts"
         / "run_action_v4_batched.py", frozen / "run_action_v4_batched.py")

    index = {}
    for path in sorted(RELEASE.rglob("*")):
        if path.is_file() and path.name not in {"sha256.json", "README.md", "DATA_LICENSE.md"}:
            index[str(path.relative_to(RELEASE))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
    (RELEASE / "sha256.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"files": len(index), "bytes": sum(x["bytes"] for x in index.values())}))


if __name__ == "__main__":
    main()
