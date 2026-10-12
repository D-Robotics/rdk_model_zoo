#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Verify compiled artifacts (and exported graphs) against the descriptor.

Compares the SHA-256 of the PartA/PartB artifacts for one target with
``model/frcnn_mbv3_320.json`` — the exact files this sample was validated
with.  The default ``--target`` selection runs only the recorded binaries;
artifacts you compiled yourself whose hashes differ still run when passed
explicitly with ``--parta-path``/``--partb-path`` (their hashes are
reported either way).  The printed verdict says which situation you are in.

Exit codes: 0 both artifacts match the descriptor; 1 an artifact is missing;
2 artifacts exist but at least one hash differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SAMPLE_MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
DESCRIPTOR_PATH = SAMPLE_MODEL_DIR / "frcnn_mbv3_320.json"
SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(target: str, parta: Path, partb: Path) -> int:
    descriptor = json.loads(DESCRIPTOR_PATH.read_text())
    if target not in descriptor["targets"]:
        print(f"error: descriptor has no target {target!r}")
        return 1
    row = descriptor["targets"][target]

    status = 0
    for role, path, expected in (
        ("parta", parta, row["parta"]["sha256"]),
        ("partb", partb, row["partb"]["sha256"]),
    ):
        if not path.is_file():
            print(f"{target}/{role}: MISSING ({path})")
            status = max(status, 1)
            continue
        digest = sha256_file(path)
        verdict = "MATCH" if digest == expected else "DIFFERS"
        print(f"{target}/{role}: {verdict}  {path}")
        print(f"  compiled  sha256 {digest}")
        print(f"  recorded  sha256 {expected}")
        if verdict == "DIFFERS":
            status = max(status, 2)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=SUPPORTED_TARGETS,
                        help="board target whose pair is verified")
    here = Path(__file__).resolve().parent
    parser.add_argument("--parta", type=Path, default=None,
                        help="PartA artifact (default: build/<target>/parta/frcnn_mbv3_320_parta.{bin,hbm})")
    parser.add_argument("--partb", type=Path, default=None,
                        help="PartB artifact (default: build/<target>/partb/frcnn_mbv3_320_partb.{bin,hbm})")
    args = parser.parse_args()

    ext = ".bin" if args.target == "x5" else ".hbm"
    parta = args.parta or here / "build" / args.target / "parta" / f"frcnn_mbv3_320_parta{ext}"
    partb = args.partb or here / "build" / args.target / "partb" / f"frcnn_mbv3_320_partb{ext}"

    onnx_dir = here / "onnx"
    if onnx_dir.is_dir():
        for name, expected in descriptor_onnx(onnx_dir):
            path = onnx_dir / name
            if path.is_file():
                digest = sha256_file(path)
                print(f"onnx/{name}: {'MATCH' if digest == expected else 'DIFFERS'}")
    return verify(args.target, parta, partb)


def descriptor_onnx(onnx_dir: Path) -> list[tuple[str, str]]:
    data = json.loads(DESCRIPTOR_PATH.read_text())
    names = {
        "part_a_f0only.op11.onnx": data["onnx"].get("parta_f0only_op11_sha256"),
        "part_a_f0only.op16.onnx": data["onnx"].get("parta_f0only_op16_sha256"),
        "part_b_batch150.op11.onnx": data["onnx"].get("partb_batch150_op11_sha256"),
        "part_b_packed.op16.onnx": data["onnx"].get("partb_packed_op16_sha256"),
    }
    return [(name, digest) for name, digest in names.items() if digest]


if __name__ == "__main__":
    sys.exit(main())
