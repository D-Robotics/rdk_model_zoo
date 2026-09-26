# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run the five source cases using the canonical single-case CLI."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import sha256_file
from samples.vision.diffusiondrive.runtime.python import main as single
from samples.vision.diffusiondrive.runtime.python.model_binding import (
    SAMPLE_DIR,
    resolve_selection,
)
from samples.vision.diffusiondrive.runtime.python.data_io import load_features

CASES = ("case_000", "case_017", "case_042", "case_073", "case_099")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    single.add_runtime_arguments(p)
    p.add_argument("--cases-root", type=Path, default=SAMPLE_DIR / "test_data")
    p.add_argument("--output", type=Path, default=Path("outputs/diffusiondrive_cases"))
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return single.main(["--target", args.target, "--list-models"])
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        # Reuse the single-case validation without executing or creating files.
        base = [
            "--target",
            selection.target,
            "--asset-id",
            selection.asset.reference,
            "--priority",
            str(args.priority),
            "--bpu-cores",
            *[str(c) for c in args.bpu_cores],
            "--agent-score-thres",
            str(args.agent_score_thres),
        ]
        if args.model_path is not None:
            base += ["--model-path", str(selection.model_path.resolve())]
        if (
            not 0 <= args.priority <= 255
            or any(c < 0 for c in args.bpu_cores)
            or not 0 <= args.agent_score_thres <= 1
        ):
            raise ValueError("Invalid scheduling or agent threshold")
        output = args.output.expanduser().resolve()
        cases_root = args.cases_root.expanduser().resolve()
        cases = []
        for case in CASES:
            inp = cases_root / case / "inputs.npz"
            load_features(inp)
            cases.append(
                {
                    "case": case,
                    "input_sha256": sha256_file(inp),
                    "argv": base
                    + ["--input-npz", str(inp), "--output", str(output / case)],
                }
            )
        record = {
            "schema_version": "1.0",
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "cases": cases,
            "runs": [],
            "status": "not-run",
            "remaining_cases": list(CASES),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.dry_run:
            print(json.dumps(record, indent=2))
            return 0
        if output.exists():
            raise FileExistsError(f"Batch output must be new: {output}")
        output.mkdir(parents=True)
        for index, case in enumerate(cases):
            rc = single.main(case["argv"])
            run = {"case": case["case"], "returncode": rc}
            evidence = output / case["case"] / "report.json"
            if evidence.is_file():
                run["report_sha256"] = sha256_file(evidence)
            record["runs"].append(run)
            record.update(
                status=(
                    "failed"
                    if rc
                    else ("completed" if index == len(cases) - 1 else "running")
                ),
                remaining_cases=list(CASES[index + 1 :]),
                updated_utc=datetime.now(timezone.utc).isoformat(),
            )
            (output / "batch-report.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            if rc:
                return rc
        print(f"Saved five case records to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
