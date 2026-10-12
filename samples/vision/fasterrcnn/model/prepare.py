#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare the compiled PartA/PartB model pairs under ``model/<target>/``.

Without ``--from-build`` the tool reports, for every requested target,
whether the installed pair is present and whether its SHA-256 digests match
``model/frcnn_mbv3_320.json``.  With ``--from-build DIR`` it installs the
pairs compiled by the conversion guide from a build directory and reports
the same comparison for the installed bytes.

Exit codes: 0 every requested pair is installed and matches the descriptor;
1 an artifact is missing (nothing installed for that target); 2 a pair is
installed but at least one digest differs from the descriptor.

Usage (from the repository root, or anywhere with the repository on
``sys.path``)::

    python samples/vision/fasterrcnn/model/prepare.py                     # status, all targets
    python samples/vision/fasterrcnn/model/prepare.py --target s100 s600  # status, subset
    python samples/vision/fasterrcnn/model/prepare.py --target x5 --from-build /path/to/build
    python samples/vision/fasterrcnn/model/prepare.py --from-build samples/vision/fasterrcnn/conversion
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python.cli import (  # noqa: E402
    SUPPORTED_TARGETS,
    descriptor_pairs,
)

MODEL_DIR = Path(__file__).resolve().parent

# Repository-relative paths used in the printed run commands.
RUN_PY = "samples/vision/fasterrcnn/runtime/python/main.py"


def sha256_file(path: Path) -> str:
    """Stream a file's sha256 hex digest (artifacts reach ~30 MB).

    Standard library only: the quickstart compiles inside Docker but runs
    this tool on the plain host outside it, so preparing models must not
    import the runtime (whose ``detect`` module needs NumPy/OpenCV).
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_pairs(build_dir: Path, target: str, pair, single: bool) -> list[tuple[Path, Path]]:
    """Source locations for one target's pair, most specific first.

    Each entry is a (PartA, PartB) candidate; the first entry where both
    halves exist wins.  The first four entries match the conversion guide's
    ``build/<target>/{parta,partb}/`` layout whether ``build_dir`` is the
    build directory itself or the ``conversion/`` directory containing it;
    the last accepts a flat pair directory for single-target requests.
    """
    pa, pb = pair.parta_file, pair.partb_file
    candidates = [
        (build_dir / target / "parta" / pa, build_dir / target / "partb" / pb),
        (build_dir / target / pa, build_dir / target / pb),
        (build_dir / "build" / target / "parta" / pa,
         build_dir / "build" / target / "partb" / pb),
        (build_dir / "build" / target / pa, build_dir / "build" / target / pb),
    ]
    if single:
        candidates.append((build_dir / pa, build_dir / pb))
    return candidates


def locate_pair(build_dir: Path, target: str, pair, single: bool) -> tuple[Path, Path] | None:
    """Return the first candidate location with both halves, else ``None``.

    A location with exactly one half is never used: the pair is installed
    together or not at all.
    """
    for parta, partb in candidate_pairs(build_dir, target, pair, single):
        if parta.is_file() and partb.is_file():
            return parta, partb
    return None


def install_pair(target: str, pair, parta_src: Path, partb_src: Path,
                 force: bool, dry_run: bool) -> bool:
    """Copy one pair into ``model/<target>/``; return True when installed.

    A half whose source already is the installed file, or whose bytes are
    already installed, is left untouched — so installing a pair from
    ``model/<target>/`` itself, or re-running an install, never rewrites
    anything.  Overwriting a differing file requires ``force``.
    """
    parta_dst = MODEL_DIR / target / pair.parta_file
    partb_dst = MODEL_DIR / target / pair.partb_file
    # Per half: "same-path" (source is the installed file), "identical"
    # (installed bytes already match) or "copy".
    states: list[str] = []
    for src, dst in ((parta_src, parta_dst), (partb_src, partb_dst)):
        if src.resolve() == dst.resolve():
            states.append("same-path")
            continue
        if dst.is_file():
            if sha256_file(dst) == sha256_file(src):
                states.append("identical")
                continue
            if not force:
                print(f"{target}: refusing to overwrite {dst} (differs from "
                      f"{src}); pass --force to replace it")
                return False
        states.append("copy")
    if dry_run:
        for (src, dst), state in zip(
                ((parta_src, parta_dst), (partb_src, partb_dst)), states):
            if state == "copy":
                print(f"{target}: would install {dst} <- {src}")
            elif state == "same-path":
                print(f"{target}: {dst} is already the installed file")
            else:
                print(f"{target}: {dst} already matches {src}")
        return True
    MODEL_DIR.joinpath(target).mkdir(parents=True, exist_ok=True)
    for (src, dst), state in zip(
            ((parta_src, parta_dst), (partb_src, partb_dst)), states):
        if state == "copy":
            shutil.copy2(src, dst)
            print(f"{target}: installed {dst} <- {src}")
        elif state == "same-path":
            print(f"{target}: {dst} is already the installed file")
        else:
            print(f"{target}: {dst} already matches {src} (left in place)")
    return True


def report_pair(target: str, pair) -> int:
    """Print the installed pair's digest comparison; return the exit code.

    A matching pair runs with plain ``--target <target>``; a differing pair
    (typically self-compiled binaries) must be passed explicitly.
    """
    status = 0
    parta_dst = MODEL_DIR / target / pair.parta_file
    partb_dst = MODEL_DIR / target / pair.partb_file
    for role, dst, expected in (("parta", parta_dst, pair.parta_sha256),
                                ("partb", partb_dst, pair.partb_sha256)):
        if not dst.is_file():
            print(f"{target}/{role}: MISSING ({dst})")
            status = max(status, 1)
            continue
        digest = sha256_file(dst)
        if digest == expected:
            print(f"{target}/{role}: MATCH {dst}")
        else:
            print(f"{target}/{role}: DIFFERS {dst}")
            print(f"  installed sha256 {digest}")
            print(f"  recorded  sha256 {expected}")
            status = max(status, 2)
    if status == 0:
        print(f"{target}: pair matches the descriptor; runs with --target {target}")
    elif status == 2:
        print(f"{target}: pair differs from the descriptor; run it explicitly:")
        print(f"  python {RUN_PY} --target {target} "
              f"--parta-path {parta_dst} --partb-path {partb_dst} -i <image>")
        print(f"  fasterrcnn --target {target} "
              f"--parta-path {parta_dst} --partb-path {partb_dst} -i <image>")
    else:
        print(f"{target}: compile the pair with the conversion guide "
              f"(samples/vision/fasterrcnn/conversion), then install it with "
              f"--from-build")
    return status


def main(argv=None) -> int:
    """Run the status report or the build-directory install."""
    parser = argparse.ArgumentParser(
        prog="prepare",
        description="Prepare compiled PartA/PartB pairs under model/<target>/")
    parser.add_argument("--target", nargs="+", choices=SUPPORTED_TARGETS,
                        default=list(SUPPORTED_TARGETS),
                        help="board target(s) to prepare (default: all)")
    parser.add_argument("--from-build", default=None, type=Path,
                        help="install pairs from a conversion build directory")
    parser.add_argument("--force", action="store_true",
                        help="overwrite installed artifacts that differ")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the planned install without writing")
    args = parser.parse_args(argv)

    _single = len(args.target) == 1
    pairs = descriptor_pairs()
    overall = 0
    for target in args.target:
        pair = pairs[target]
        if args.from_build is None:
            overall = max(overall, report_pair(target, pair))
            continue
        located = locate_pair(args.from_build, target, pair, single=_single)
        if located is None:
            print(f"{target}: no pair found under {args.from_build} "
                  f"(expected {pair.parta_file} and {pair.partb_file} in "
                  f"{args.from_build}/{target}/parta|partb/)")
            overall = max(overall, 1)
            continue
        if not install_pair(target, pair, *located, force=args.force,
                            dry_run=args.dry_run):
            overall = max(overall, 1)
            continue
        if not args.dry_run:
            overall = max(overall, report_pair(target, pair))
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
