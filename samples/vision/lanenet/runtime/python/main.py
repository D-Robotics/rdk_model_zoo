# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""LaneNet CLI: preparation-free inference, raw evidence and separate displays."""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.lanenet.runtime.python.model_binding import (
    SAMPLE_DIR,
    resolve_selection,
    list_available_assets,
)
from samples.vision.lanenet.runtime.python.model_runner import RuntimeModelRunner


def build_parser():
    p = argparse.ArgumentParser(
        description="LaneNet embeddings and binary labels; S100 published asset only"
    )
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--test-img", type=Path, default=SAMPLE_DIR / "test_data/lane.jpg")
    p.add_argument("--output", type=Path, default=Path("outputs/lanenet"))
    p.add_argument(
        "--instance-save-path",
        type=Path,
        help="Optional additional embedding display, not clustered lane IDs",
    )
    p.add_argument(
        "--binary-save-path", type=Path, help="Optional additional binary display"
    )
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
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
        if not 0 <= args.priority <= 255 or any(core < 0 for core in args.bpu_cores):
            raise ValueError("priority must be 0..255 and cores nonnegative")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "model_path": str(selection.model_path),
                        "sdk_loaded": False,
                        "downloaded": False,
                        "clustering_performed": False,
                    },
                    indent=2,
                )
            )
            return 0
        output = args.output.expanduser()
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        extras = {
            name: path.expanduser()
            for name, path in [
                ("instance", args.instance_save_path),
                ("binary", args.binary_save_path),
            ]
            if path is not None
        }
        reserved = {
            (output / name).resolve()
            for name in (
                "raw_outputs.npz",
                "embedding.npy",
                "binary.npy",
                "instance_pred.png",
                "binary_pred.png",
                "report.json",
            )
        }
        destinations = [p.resolve() for p in extras.values()]
        if len(set(destinations)) != len(destinations) or any(
            p in reserved for p in destinations
        ):
            raise ValueError(
                "Additional displays must have distinct paths outside canonical output filenames"
            )
        for path in extras.values():
            if path.exists():
                raise FileExistsError(f"Use a new additional image path: {path}")
        import cv2
        import numpy as np
        from samples._shared.assets import sha256_file
        from samples._shared.runtime_meta import metadata_evidence
        from samples.vision.lanenet.runtime.python.lanenet import LaneNetTask
        from samples.vision.lanenet.runtime.python.visualization import (
            embedding_image,
            binary_image,
        )

        image_path = args.test_img.expanduser()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = LaneNetTask(runner, binding)
        raw = task.forward(task.pre_process(image))
        result = task.post_process(raw)
        displays = {
            "instance": embedding_image(result.embedding),
            "binary": binary_image(result.binary),
        }
        # Fixed archive keys avoid arbitrary SDK names colliding with np.savez
        # keyword parameters. The complete name-to-key relation stays explicit.
        keys = {
            name: f"output_{index}"
            for index, name in enumerate(binding.metadata.output_names)
        }
        report = {
            "schema_version": "1.0",
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "model_path": str(selection.model_path),
            "model_sha256": sha256_file(selection.model_path),
            "publisher_sha256": selection.asset.sha256,
            "input": str(image_path),
            "input_sha256": sha256_file(image_path),
            "input_shape": list(image.shape),
            "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
            "runtime_metadata": metadata_evidence(binding.metadata),
            "raw_tensor_keys": keys,
            "embedding_shape": list(result.embedding.shape),
            "binary_shape": list(result.binary.shape),
            "clustering_performed": False,
            "output_grid": "model 256x512; not original image size",
            "embedding_display": "clip to [0,1], multiply255, round ties-to-even, preserve channel order",
            "binary_display": "validated 0/1 labels times255",
            "priority": args.priority,
            "bpu_cores": args.bpu_cores,
            "additional_images": {name: str(path) for name, path in extras.items()},
            "latency": "not measured",
        }
        output.mkdir(parents=True, exist_ok=False)
        np.savez(
            output / "raw_outputs.npz",
            **{keys[name]: value for name, value in raw.items()},
        )
        np.save(output / "embedding.npy", result.embedding)
        np.save(output / "binary.npy", result.binary)
        writes = [
            (output / "instance_pred.png", displays["instance"]),
            (output / "binary_pred.png", displays["binary"]),
        ] + [(path, displays[name]) for name, path in extras.items()]
        for path, value in writes:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), value):
                raise OSError(f"Failed to save image: {path}")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved model-grid embedding, binary labels and provenance to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
