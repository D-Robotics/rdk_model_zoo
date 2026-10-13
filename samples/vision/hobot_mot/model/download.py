"""Fetch and verify the pinned upstream HobotMot (D-Robotics/mot) source bundle.

The tracker runs on the CPU, so the preparation step fetches upstream source
(wrappers compiled by this sample plus the prebuilt ``iou_based_mot`` core and
its dependency archives) instead of a neural model. Binaries never enter git.
"""
import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

UPSTREAM_COMMIT = "0120205146d88c3474bfc2936522e349e169fa8d"
UPSTREAM_URL = (
    "https://codeload.github.com/D-Robotics/mot/tar.gz/" + UPSTREAM_COMMIT
)
# sha256 of the GitHub-generated tarball for the pinned commit.
UPSTREAM_SHA256 = "a0bf3b9738af2f7c4be5885ac777cdf93dab6d4dd374a297ceebebe58268b54b"

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
BUNDLE_PATH = VENDOR_DIR / f"mot-{UPSTREAM_COMMIT}.tar.gz"
SOURCE_DIR = VENDOR_DIR / f"mot-{UPSTREAM_COMMIT}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(force: bool = False) -> Path:
    """Download, checksum and extract the pinned upstream source.

    Args:
        force: re-download even if the bundle is already present.

    Returns:
        Path of the extracted upstream source tree.

    Raises:
        OSError: download, checksum or extraction failure.
    """
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    if force or not BUNDLE_PATH.exists():
        print(f"Downloading {UPSTREAM_URL}")
        urllib.request.urlretrieve(UPSTREAM_URL, BUNDLE_PATH)
    observed = sha256_file(BUNDLE_PATH)
    if observed != UPSTREAM_SHA256:
        BUNDLE_PATH.unlink(missing_ok=True)
        raise OSError(
            f"checksum mismatch for {BUNDLE_PATH.name}: expected "
            f"{UPSTREAM_SHA256}, observed {observed}"
        )
    print(f"SHA-256 OK: {observed}")
    if not SOURCE_DIR.exists():
        with tarfile.open(BUNDLE_PATH, "r:gz") as archive:
            root = archive.getnames()[0].split("/")[0]
            if root != f"mot-{UPSTREAM_COMMIT}":
                raise OSError(f"unexpected archive root {root!r}")
            archive.extractall(VENDOR_DIR)
    print(f"Upstream source ready: {SOURCE_DIR}")
    return SOURCE_DIR


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download the source bundle"
    )
    args = parser.parse_args(argv)
    try:
        prepare(force=args.force)
    except (OSError, tarfile.TarError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
