# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pinned Google base224 preprocessing, independent of legacy letterboxing."""
from pathlib import Path
import numpy as np
from PIL import Image


def prepare_image(source):
    """Return RGB F32 NCHW using upstream bicubic direct resize and 0.5 stats."""
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            rgb = image.convert('RGB')
    elif isinstance(source, Image.Image):
        rgb = source.convert('RGB')
    else:
        raise TypeError('Expected an image path or PIL image')
    pixels = np.asarray(rgb.resize((224, 224), Image.Resampling.BICUBIC))
    # Match the upstream rescale's F64 multiply followed by an F32 cast.
    values = (pixels * (1 / 255)).astype(np.float32)
    values = (values - np.float32(.5)) / np.float32(.5)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])
