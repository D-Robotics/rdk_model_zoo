# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Verify the locally prepared LiteAvatar model package against the manifest.

Manual-availability packages are not downloaded by any script.  This helper
checks one target's four HBM artifacts (plus the bundled CMVN file) under
this directory: pick the target with ``--target`` (s100, s100p or s600 —
each set lives under its own ``model/<target>/`` subdirectory), and for each
published filename it reports whether the file is present and whether its
SHA-256 matches the manifest digest, so a transferred golden bundle can be
validated before the runtime is started.  The manifest must publish exactly
the four stages of that target — a partial set (or rows beyond the four) is
rejected as a manifest defect instead of verifying a subset.  A locally
rebuilt set is expected to differ; identify it with the runtime's
``--models-manifest`` option instead of this golden check.

Exit codes: 0 all present and matching; 1 at least one missing or mismatched;
2 usage error.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.lite_avatar.runtime.python.cli import (  # noqa: E402
    FILENAMES as TARGET_FILENAMES,
    STAGES,
)
from utils.py_utils.assets import list_assets, sha256_file  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent

TARGETS = ("s100", "s100p", "s600")

CMVN_NAME = "am.mvn"
CMVN_SHA256 = "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS, default="s100")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory holding the manifest-relative model paths.",
    )
    args = parser.parse_args(argv)
    root = args.model_dir.expanduser()
    prefix = f"{args.target}/"

    failures = []
    print(f"LiteAvatar {args.target} package under {root}:")
    assets = [
        asset
        for asset in list_assets("s", "lite_avatar")
        if asset.filename.startswith(prefix)
    ]
    if not assets:
        print(f"  no manifest assets published for target {args.target!r}")
        return 2
    expected = TARGET_FILENAMES[args.target]
    by_name = {}
    for asset in assets:
        by_name.setdefault(asset.filename, []).append(asset)
    missing = [expected[stage] for stage in STAGES if expected[stage] not in by_name]
    extra = sorted(set(by_name) - set(expected.values()))
    duplicates = sorted(name for name, rows in by_name.items() if len(rows) > 1)
    if missing or extra or duplicates:
        for name in missing:
            print(f"  MANIFEST DEFECT: stage not published for {args.target}: {name}")
        for name in extra:
            print(f"  MANIFEST DEFECT: row beyond the four {args.target} stages: {name}")
        for name in duplicates:
            print(f"  MANIFEST DEFECT: duplicate manifest row: {name}")
        print(
            f"  verification requires exactly the four {args.target} stages: "
            + ", ".join(expected[stage] for stage in STAGES)
        )
        return 2
    for stage in STAGES:
        asset = by_name[expected[stage]][0]
        path = root / asset.filename
        if not path.is_file():
            print(f"  MISSING  {asset.filename} (sha256 {asset.sha256})")
            failures.append(asset.filename)
            continue
        observed = sha256_file(path)
        state = "OK" if observed == asset.sha256 else "MISMATCH"
        print(f"  {state:8}{asset.filename}")
        if state != "OK":
            print(f"           manifest sha256 {asset.sha256}")
            print(f"           observed sha256 {observed}")
            failures.append(asset.filename)
    cmvn = root / CMVN_NAME
    if cmvn.is_file():
        state = "OK" if sha256_file(cmvn) == CMVN_SHA256 else "MISMATCH"
        print(f"  {state:8}{CMVN_NAME} (bundled CMVN)")
        if state != "OK":
            failures.append(CMVN_NAME)
    else:
        print(f"  MISSING  {CMVN_NAME} (bundled CMVN)")
        failures.append(CMVN_NAME)

    if failures:
        print(
            f"{len(failures)} file(s) missing or mismatched; prepare per "
            "model/README.md (golden bundle) or conversion/README.md "
            "(self-compilation, then identify with --models-manifest)."
        )
        return 1
    print(f"All {args.target} artifacts match the published digests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
