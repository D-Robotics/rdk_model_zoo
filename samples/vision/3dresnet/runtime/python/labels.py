# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Independent Kinetics-400 label-file decoding."""
from __future__ import annotations

import json
from pathlib import Path


def load_labels(path: str | Path) -> dict[int, str]:
    with Path(path).expanduser().open(encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict) or len(values) != 400:
        raise ValueError("Expected a 400-entry Kinetics class-name mapping.")
    result: dict[int, str] = {}
    for name, class_id in values.items():
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise ValueError(f"Invalid Kinetics class id: {class_id!r}; expected JSON integer.")
        index = class_id
        if index in result or not 0 <= index < 400:
            raise ValueError(f"Kinetics class ids must be unique integers in [0,399], got {index}.")
        result[index] = str(name).replace('"', "")
    if set(result) != set(range(400)):
        raise ValueError("Kinetics class ids must cover every id from 0 through 399.")
    return result


__all__ = ["load_labels"]
