"""Download pinned Hugging Face model files without a repo-tree API request."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("revision must be a full commit SHA")

    def download(name: str) -> Path:
        return Path(hf_hub_download(args.repo, name, revision=args.revision,
                                    cache_dir=args.cache_dir))

    required = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "model.safetensors.index.json")
    paths = {name: download(name) for name in required}
    try:
        paths["generation_config.json"] = download("generation_config.json")
    except EntryNotFoundError:
        pass
    index = json.loads(paths["model.safetensors.index.json"].read_text())
    shards = sorted(set(index["weight_map"].values()))
    with ThreadPoolExecutor(max_workers=min(4, len(shards))) as pool:
        for name, path in zip(shards, pool.map(download, shards)):
            paths[name] = path
    print(json.dumps({"repo": args.repo, "revision": args.revision,
                      "snapshot": str(paths["config.json"].parent),
                      "files": sorted(paths),
                      "weight_bytes": sum(paths[name].stat().st_size for name in shards)},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
