# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Fetch MobileSAM source and checkpoint for the X5 conversion recipe."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

TARGETS = ("x5", "s100", "s100p", "s600")
REPO_URL = "https://github.com/ChaoningZhang/MobileSAM.git"
CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MobileSAM upstream source and checkpoint.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"))
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--checkpoint-url", default=CHECKPOINT_URL)
    return parser


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.target != "x5":
        raise SystemExit("The fixed S source provides no download_assets.py; prepare the upstream checkout and checkpoint manually.")
    args.workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = args.workspace / "MobileSAM"
    if not repo_dir.exists():
        run(["git", "clone", args.repo_url, str(repo_dir)])
    else:
        print(f"Reuse existing repository: {repo_dir}")
    weight_dir = repo_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / "mobile_sam.pt"
    if not weight_path.exists() or weight_path.stat().st_size < 1_000_000:
        print(f"Downloading {args.checkpoint_url} -> {weight_path}")
        urlretrieve(args.checkpoint_url, weight_path)
    print(f"Repository: {repo_dir}\nCheckpoint: {weight_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
