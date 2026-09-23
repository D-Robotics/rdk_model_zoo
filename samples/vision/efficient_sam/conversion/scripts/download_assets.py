# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Fetch EfficientSAM source and checkpoint for the X5 conversion recipe."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

TARGETS = ("x5", "s100", "s100p", "s600")
REPO_URL = "https://github.com/yformer/EfficientSAM.git"
ZIP_URL = "https://github.com/yformer/EfficientSAM/archive/refs/heads/main.zip"
CHECKPOINT_URL = "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare EfficientSAM upstream source and checkpoint.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"))
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--zip-url", default=ZIP_URL)
    parser.add_argument("--checkpoint-url", default=CHECKPOINT_URL)
    return parser


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _safe_extract(zip_path: Path, workspace: Path) -> Path:
    """Extract only an EfficientSAM-main tree in a private staging directory."""
    workspace = workspace.resolve()
    staging = Path(tempfile.mkdtemp(prefix=".efficient-sam-extract-", dir=workspace))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                parts = Path(member.filename).parts
                if not parts or parts[0] != "EfficientSAM-main" or ".." in parts:
                    raise RuntimeError(f"Archive member is outside EfficientSAM-main: {member.filename}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise RuntimeError(f"Symlink is not allowed in source archive: {member.filename}")
                target = (staging / member.filename).resolve()
                target.relative_to(staging)
            archive.extractall(staging)
        extracted = staging / "EfficientSAM-main"
        if not extracted.is_dir():
            raise FileNotFoundError(f"Archive did not contain {extracted}")
        return extracted
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.target != "x5":
        raise SystemExit("The fixed S source provides no download_assets.py; prepare the upstream checkout and checkpoint manually.")
    args.workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = args.workspace / "EfficientSAM"
    if not repo_dir.exists():
        clone_staging = Path(tempfile.mkdtemp(prefix=".efficient-sam-clone-", dir=args.workspace))
        clone_repo = clone_staging / "EfficientSAM"
        try:
            run(["git", "clone", args.repo_url, str(clone_repo)])
            clone_repo.rename(repo_dir)
        except Exception:
            shutil.rmtree(clone_staging, ignore_errors=True)
            zip_path = args.workspace / "EfficientSAM-main.zip"
            print(f"git clone failed; downloading source zip: {args.zip_url}")
            urlretrieve(args.zip_url, zip_path)
            extracted = _safe_extract(zip_path, args.workspace)
            staging = extracted.parent
            try:
                extracted.rename(repo_dir)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
    weight_dir = repo_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / "efficient_sam_vitt.pt"
    if not weight_path.exists() or weight_path.stat().st_size < 10_000_000:
        print(f"Downloading {args.checkpoint_url} -> {weight_path}")
        urlretrieve(args.checkpoint_url, weight_path)
    print(f"Repository: {repo_dir}\nCheckpoint: {weight_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
