# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Strict NPZ loading and non-overwriting output paths, independent of inference."""

from pathlib import Path
import numpy as np
from samples.vision.diffusiondrive.runtime.python.model_binding import INPUT_SHAPES


def load_npz(path):
    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError("Expected an NPZ archive")
    with archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate NPZ names are ambiguous")
        return {name: archive[name].copy() for name in archive.files}


def load_features(path):
    values = load_npz(path)
    if set(values) != set(INPUT_SHAPES):
        raise ValueError("Feature archive requires camera/lidar/status/noise only")
    for name, shape in INPUT_SHAPES.items():
        value = values[name]
        if (
            value.shape != shape
            or value.dtype != np.dtype("float32")
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid logical feature {name}: expected finite float32 {shape}"
            )
    return values


def validate_destinations(output, extras):
    output = Path(output).expanduser().resolve()
    paths = [Path(p).expanduser().resolve() for p in extras]
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    reserved = {
        output / name
        for name in (
            "raw_outputs.npz",
            "physical_inputs.npz",
            "outputs.npz",
            "result.png",
            "report.json",
            "batch-report.json",
        )
    }
    if len(set(paths)) != len(paths):
        raise ValueError("Additional output paths must be distinct")
    for p in paths:
        if p.exists() or p == output or p in output.parents or p in reserved:
            raise ValueError(f"Additional output path exists or conflicts: {p}")
    return output
