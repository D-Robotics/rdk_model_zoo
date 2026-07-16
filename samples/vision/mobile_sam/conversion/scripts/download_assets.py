"""Download MobileSAM source and checkpoint for ONNX export.

This script intentionally keeps the upstream repository outside this sample.
It records the exact source location used by the conversion scripts.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

REPO_URL = "https://github.com/ChaoningZhang/MobileSAM.git"
CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run an external command and echo it before execution.

    Args:
        cmd: Command and arguments to execute.
        cwd: Optional working directory for the command.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
    """

    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    """Download the upstream MobileSAM repository and checkpoint."""

    parser = argparse.ArgumentParser(description="Download MobileSAM source and checkpoint.")
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"), help="Directory used for upstream source and weights.")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = args.workspace / "MobileSAM"
    if not repo_dir.exists():
        run(["git", "clone", REPO_URL, str(repo_dir)])
    else:
        print(f"Reuse existing repository: {repo_dir}")
    weight_dir = repo_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / "mobile_sam.pt"
    if not weight_path.exists() or weight_path.stat().st_size < 1_000_000:
        print(f"Downloading {CHECKPOINT_URL} -> {weight_path}")
        urlretrieve(CHECKPOINT_URL, weight_path)
    print("MobileSAM repo:", repo_dir)
    print("Checkpoint:", weight_path)


if __name__ == "__main__":
    main()