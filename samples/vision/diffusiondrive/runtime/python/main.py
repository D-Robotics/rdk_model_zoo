# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run one deterministic planning example with raw IO and full provenance."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import sha256_file
from samples._shared.runtime_meta import metadata_evidence
from samples.vision.diffusiondrive.runtime.python.model_binding import (
    SAMPLE_DIR,
    resolve_selection,
    list_available_assets,
)
from samples.vision.diffusiondrive.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.diffusiondrive.runtime.python.data_io import (
    load_features,
    validate_destinations,
)
from samples.vision.diffusiondrive.runtime.python.diffusiondrive import (
    DiffusionDriveTask,
)


def add_runtime_arguments(p):
    p.add_argument(
        "--target",
        "--platform",
        dest="target",
        choices=("auto", "x5", "s100", "s100p", "s600"),
        default="auto",
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--agent-score-thres", type=float, default=0.5)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(p)
    p.add_argument(
        "--input-npz", type=Path, default=SAMPLE_DIR / "test_data/reference_inputs.npz"
    )
    p.add_argument("--output", type=Path, default=Path("outputs/diffusiondrive"))
    p.add_argument(
        "--output-npz",
        type=Path,
        help="Optional additional decoded archive; canonical outputs.npz is always retained",
    )
    p.add_argument(
        "--img-save-path",
        "--output-image",
        dest="img_save_path",
        type=Path,
        help="Optional additional visualization",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(
                json.dumps(
                    [
                        {"asset_id": a.reference, "url": a.url, "sha256": a.sha256}
                        for a in list_available_assets(args.target)
                    ],
                    indent=2,
                )
            )
            return 0
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        if (
            not np.isfinite(args.agent_score_thres)
            or not 0 <= args.agent_score_thres <= 1
        ):
            raise ValueError("Agent score threshold must be finite and within [0,1]")
        if not 0 <= args.priority <= 255 or any(c < 0 for c in args.bpu_cores):
            raise ValueError("Priority must be 0..255; core IDs must be nonnegative")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "model_path": str(selection.model_path),
                        "input_npz": str(args.input_npz),
                        "sdk_loaded": False,
                        "downloaded": False,
                    },
                    indent=2,
                )
            )
            return 0
        extras = {
            key: p.expanduser().resolve()
            for key, p in (("decoded", args.output_npz), ("image", args.img_save_path))
            if p is not None
        }
        output = validate_destinations(args.output, extras.values())
        if "image" in extras and extras["image"].suffix.lower() not in (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
        ):
            raise ValueError("Additional image requires PNG/JPEG/BMP extension")
        input_path = args.input_npz.expanduser().resolve()
        features = load_features(input_path)
        started = datetime.now(timezone.utc).isoformat()
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = DiffusionDriveTask(runner, binding, args.agent_score_thres)
        physical = task.pre_process(features)
        raw = task.forward(physical)
        decoded = task.post_process(raw)
        from samples.vision.diffusiondrive.runtime.python.visualization import (
            render_result,
        )
        import cv2

        canvas = render_result(features, decoded, selection.target.upper())
        report = {
            "schema_version": "1.0",
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "model_path": str(selection.model_path.resolve()),
            "model_sha256": sha256_file(selection.model_path),
            "publisher_sha256": selection.asset.sha256,
            "input_npz": str(input_path),
            "input_sha256": sha256_file(input_path),
            "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
            "runtime_metadata": metadata_evidence(binding.metadata),
            "scheduling": {"priority": args.priority, "bpu_cores": args.bpu_cores},
            "agent_score_threshold": args.agent_score_thres,
            "started_utc": started,
            "finished_inference_utc": datetime.now(timezone.utc).isoformat(),
            "noise": "fixed caller-supplied tensor; no regeneration",
            "latency": "not measured",
            "actuation": False,
            "additional_outputs": {k: str(p) for k, p in extras.items()},
        }
        output.mkdir(parents=True, exist_ok=False)
        np.savez(output / "physical_inputs.npz", **physical)
        np.savez(output / "raw_outputs.npz", **raw)
        np.savez(output / "outputs.npz", **decoded)
        if not cv2.imwrite(str(output / "result.png"), canvas):
            raise OSError("Failed to save visualization")
        for key, path in extras.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if key == "decoded":
                shutil.copyfile(output / "outputs.npz", path)
            elif not cv2.imwrite(str(path), canvas):
                raise OSError(f"Failed to save {path}")
        report["output_sha256"] = {
            name: sha256_file(output / name)
            for name in (
                "physical_inputs.npz",
                "raw_outputs.npz",
                "outputs.npz",
                "result.png",
            )
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved trajectory, agents, BEV and raw IO to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
