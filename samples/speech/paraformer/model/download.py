"""Explicitly prepare the six-file published Paraformer S100 package."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.py_utils.assets import download_asset, list_assets, sha256_file
from samples.speech.paraformer.runtime.python.pipeline import validate_vocabulary
from samples.speech.paraformer.runtime.python.cli import resolve_selections, converted_profiles, LOCAL_DIGESTS, VOCABULARY_DIGEST

MODEL_DIR = Path(__file__).resolve().parent


def _check_digest(path, expected):
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"Content differs from the pinned Paraformer package: {path}")


def _install_local(source, destination, expected):
    _check_digest(source, expected)
    if destination.exists():
        _check_digest(destination, expected)
        return expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=destination.name, suffix=".part", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(source.read_bytes())
        _check_digest(temporary, expected)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _check_digest(destination, expected)
        return expected
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("s100",), default="s100")
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--variant", help="Exact converted profile; omitted keeps the historical package")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all six sources/destinations without downloads or writes",
    )
    args = parser.parse_args(argv)
    try:
        selections = resolve_selections(args.target, variant=args.variant)
        assets = list_assets("s", "paraformer")
        expected = {s.asset.filename for s in selections} | {
            "s100/tokens.json",
            "s100/am.mvn",
            "s100/paraformer_config.yaml",
        }
        tokens_filename = "s100/tokens.json"
        if args.variant:
            entry = converted_profiles(args.target)[args.variant]
            tokens_filename = entry['tokens']
            expected = {s.asset.filename for s in selections} | {tokens_filename,
                "s100/am.mvn", "s100/paraformer_config.yaml"}
        assets = [asset for asset in assets if asset.filename in expected]
        if {asset.filename for asset in assets} != expected:
            raise ValueError("Expected the selected six-file Paraformer package")
        for asset in assets:
            name = Path(asset.filename).name
            destination = args.output_dir.expanduser() / ("s100/tokens.json" if asset.filename == tokens_filename else asset.filename)
            if args.dry_run:
                print(
                    f"{asset.reference}: {asset.url or str(MODEL_DIR / name)} -> {destination}"
                )
                continue
            if name in LOCAL_DIGESTS:
                digest = _install_local(
                    MODEL_DIR / name, destination, LOCAL_DIGESTS[name]
                )
            else:
                digest = download_asset(asset, destination)
                if asset.filename == tokens_filename:
                    _check_digest(destination, VOCABULARY_DIGEST)
                    validate_vocabulary(
                        json.loads(destination.read_text(encoding="utf-8"))
                    )
            print(f"{asset.reference}: {destination} observed-sha256={digest}")
        if not args.dry_run:
            print(
                "Package prepared. HBM hashes are observations, not publisher authentication or SDK validation."
            )
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
