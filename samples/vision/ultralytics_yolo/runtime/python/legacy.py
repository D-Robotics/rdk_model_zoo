# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Executable adapters for the former detection stage surface; no cached context."""


def pre_process_with_transform(task, image, image_format="BGR"):
    """Keep the old (tensors, transform) tuple via the canonical prepared result."""
    prepared = task.pre_process(image, image_format)
    return prepared.tensors, prepared.transform
