# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Same-board fixed-image source/unified MODNet comparison with complete evidence.

This executable runs the fixed source wrapper and the unified task on the same
image and artifact, records the prepared input tensors, raw mattes and final
uint8 mattes each side actually produced, binds the run to code/model/image
hashes, and writes a machine-readable comparison with a return code.  It never
compares hand-supplied matte files in place of a real inference and never turns a
mismatch into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples._shared.assets import verify_asset_file  # noqa: E402
from samples._shared.platforms import require_execution_target  # noqa: E402
from samples._shared.runtime_meta import RuntimeMetadata, metadata_evidence  # noqa: E402
from samples.vision.modnet.runtime.python.modnet import MODNetTask  # noqa: E402
from samples.vision.modnet.runtime.python.model_binding import (  # noqa: E402
    SAMPLE_DIR,
    resolve_selection,
)
from samples.vision.modnet.runtime.python.model_runner import RuntimeModelRunner  # noqa: E402
from samples.vision.modnet.evaluator.source_reference import (  # noqa: E402
    load_legacy,
    source_paths,
)

REF_SIZE = 512


def _hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported evidence value {type(value).__name__}.")


def _validate_scheduling(priority, bpu_cores) -> list[int]:
    if type(priority) is not int or not 0 <= priority <= 255:
        raise ValueError("priority must be 0..255.")
    if bpu_cores is not None and (
        not bpu_cores or any(type(core) is not int or core < 0 for core in bpu_cores)
    ):
        raise ValueError("Invalid BPU cores.")
    return [0] if bpu_cores is None else list(bpu_cores)


def run_comparison(selection, image, image_path, output_dir, *, priority=0, bpu_cores=None,
                   runtime_factory=None):
    """Run both implementations and retain inputs, raw mattes, result and identity.

    Requires a new output directory, a valid board identity and a readable image.
    ``runtime_factory`` is a host test seam only; the identity gate still runs.
    A failed execution leaves an error record behind and re-raises.
    """

    actual = require_execution_target(selection.target)
    image_path = Path(image_path).expanduser().resolve()
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        raise ValueError("MODNet evaluator requires a BGR HWC image.")
    if not image_path.is_file():
        raise FileNotFoundError(f"MODNet input image does not exist: {image_path}")
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"Evidence directory already exists: {directory}")
    cores = _validate_scheduling(priority, bpu_cores)

    records = {side: {} for side in ("legacy", "unified")}
    summary = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        argv=list(sys.argv),
        cwd=str(Path.cwd()),
        target=actual,
        asset_id=selection.asset.reference,
        publisher_sha256=selection.asset.sha256,
        model_path=str(Path(selection.model_path).resolve()),
        model_sha256=None,
        image_path=str(image_path),
        image_sha256=None,
        decoded_shape=list(image.shape),
        decoded_dtype=str(image.dtype),
        ref_size=REF_SIZE,
        priority=priority,
        bpu_cores=cores,
        host_versions=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__),
        metadata={},
        code_sha256={},
        passed=False,
    )
    summary["source_ref"] = "ac115717197920355fc390bb04299b20e6436864"
    summary["board_identity"] = {
        str(path): (path.read_text().strip() if path.is_file() else None)
        for path in (Path("/sys/class/boardinfo/soc_name"), Path("/sys/class/boardinfo/board_type"))
    }

    directory.mkdir(parents=True, exist_ok=False)
    failure = None
    try:
        summary["model_sha256"] = _hash(selection.model_path)
        summary["image_sha256"] = _hash(image_path)
        verify_asset_file(selection.asset, selection.model_path)
        code = list((SAMPLE_DIR / "runtime" / "python").glob("*.py"))
        code += list((SAMPLE_DIR / "evaluator").glob("*.py"))
        code += list(source_paths("x5"))
        code += [ROOT / "samples" / "_shared" / name
                 for name in ("assets.py", "platforms.py", "runtime_meta.py")]
        summary["code_sha256"] = {str(path.relative_to(ROOT)): _hash(path) for path in code}

        if runtime_factory is None:
            from samples._shared.model_runner import _default_runtime_factory

            runtime_factory = _default_runtime_factory()

        def factory(side):
            def create(path):
                runtime = runtime_factory(path)
                # Projected without copying SDK quant descriptors (asdict deepcopies and the board QuantParams refuses it).
                summary["metadata"][side] = metadata_evidence(RuntimeMetadata.from_runtime(runtime))

                class Recorder:
                    def __getattr__(self, name):
                        return getattr(runtime, name)

                    def run(self, values):
                        records[side]["inputs"] = {
                            name: array.copy()
                            for name, array in values[runtime.model_names[0]].items()
                        }
                        out = runtime.run(values)
                        records[side]["outputs"] = {
                            name: array.copy()
                            for name, array in out[runtime.model_names[0]].items()
                        }
                        return out

                return Recorder()

            return create

        legacy = load_legacy(selection, factory("legacy"), ref_size=REF_SIZE)
        legacy.set_scheduling_params(priority=priority, bpu_cores=cores)
        records["legacy"]["result"] = {"matte": np.asarray(legacy.predict(image)).copy()}

        runner = RuntimeModelRunner(selection, runtime_factory=factory("unified"))
        binding = runner.load()
        runner.set_scheduling_params(priority=priority, bpu_cores=cores)
        records["unified"]["result"] = {"matte": np.asarray(
            MODNetTask(runner, binding).predict(image)).copy()}

        checks = {}
        max_diff = {}
        for category in ("inputs", "outputs", "result"):
            left, right = records["legacy"][category], records["unified"][category]
            checks[category + "_names"] = set(left) == set(right)
            for name in set(left) & set(right):
                first, second = left[name], right[name]
                key = category + "." + name
                shape_ok = first.shape == second.shape
                dtype_ok = first.dtype == second.dtype
                finite = bool(np.isfinite(first).all() and np.isfinite(second).all())
                atol = 0 if category in ("inputs", "result") else 1e-5
                checks[key] = bool(shape_ok and dtype_ok and finite
                                   and np.allclose(first, second, rtol=0, atol=atol))
                max_diff[key] = (float(np.max(np.abs(first.astype(float) - second.astype(float))))
                                 if shape_ok and first.size else (0.0 if shape_ok else None))
        summary.update(
            checks=checks,
            max_abs_diff=max_diff,
            tolerances=dict(inputs=0, raw_matte=1e-5, result_matte=0, rtol=0),
            passed=all(checks.values()),
        )
    except Exception as exc:  # noqa: BLE001 - recorded and re-raised
        summary["error"] = dict(type=type(exc).__name__, message=str(exc))
        failure = exc
    finally:
        arrays = {}
        for side, categories in records.items():
            for category, values in categories.items():
                for index, (name, value) in enumerate(values.items()):
                    filename = f"{side}_{category}_{index}.npy"
                    np.save(directory / filename, value)
                    arrays[filename] = dict(tensor_name=name, shape=list(value.shape),
                                            dtype=str(value.dtype),
                                            sha256=_hash(directory / filename))
        summary["arrays"] = arrays
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["return_code"] = 2 if failure else (0 if summary["passed"] else 1)
        (directory / "comparison.json").write_text(
            json.dumps(summary, default=_json, allow_nan=False, indent=2) + "\n"
        )
    if failure:
        raise failure
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("x5",), required=True)
    parser.add_argument("--asset-id")
    parser.add_argument("--model-path")
    parser.add_argument("--test-img", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        image_path = args.test_img or (SAMPLE_DIR / "test_data" / "person.jpg")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        summary = run_comparison(selection, image, image_path, args.output_dir,
                                 priority=args.priority, bpu_cores=args.bpu_cores)
        print(json.dumps(dict(passed=summary["passed"], return_code=summary["return_code"],
                              evidence=str(Path(args.output_dir).resolve()))))
        return summary["return_code"]
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
