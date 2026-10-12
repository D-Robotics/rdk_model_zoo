# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Verify the locally prepared LiteAvatar model package against the manifest.

Manual-availability packages are not downloaded by any script.  This helper
checks the four S100 HBM artifacts (and the bundled CMVN file) that live under
this directory: for each published filename it reports whether the file is
present and whether its SHA-256 matches the manifest digest, so a transferred
golden bundle can be validated before the runtime is started.  A locally
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

from utils.py_utils.assets import list_assets, sha256_file  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent

CMVN_NAME = "am.mvn"
CMVN_SHA256 = "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("s100",), default="s100")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory holding the manifest-relative model paths.",
    )
    args = parser.parse_args(argv)
    root = args.model_dir.expanduser()

    failures = []
    print(f"LiteAvatar {args.target} package under {root}:")
    for asset in list_assets("s", "lite_avatar"):
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
    print("All artifacts match the published digests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
