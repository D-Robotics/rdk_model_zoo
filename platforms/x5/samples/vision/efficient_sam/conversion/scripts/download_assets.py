"""Download EfficientSAM source and checkpoint for ONNX export."""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

REPO_URL = "https://github.com/yformer/EfficientSAM.git"
ZIP_URL = "https://github.com/yformer/EfficientSAM/archive/refs/heads/main.zip"
CHECKPOINT_URL = "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vitt.pt"


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
    """Download the upstream EfficientSAM repository and checkpoint."""

    parser = argparse.ArgumentParser(description="Download EfficientSAM source and checkpoint.")
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"), help="Directory used for upstream source and weights.")
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    repo_dir = args.workspace / "EfficientSAM"
    if not repo_dir.exists():
        try:
            run(["git", "clone", REPO_URL, str(repo_dir)])
        except Exception:
            zip_path = args.workspace / "EfficientSAM-main.zip"
            print(f"git clone failed; downloading source zip: {ZIP_URL}")
            urlretrieve(ZIP_URL, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(args.workspace)
            extracted = args.workspace / "EfficientSAM-main"
            extracted.rename(repo_dir)
    weight_dir = repo_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weight_dir / "efficient_sam_vitt.pt"
    if not weight_path.exists() or weight_path.stat().st_size < 10_000_000:
        print(f"Downloading {CHECKPOINT_URL} -> {weight_path}")
        urlretrieve(CHECKPOINT_URL, weight_path)
    print("EfficientSAM repo:", repo_dir)
    print("Checkpoint:", weight_path)


if __name__ == "__main__":
    main()