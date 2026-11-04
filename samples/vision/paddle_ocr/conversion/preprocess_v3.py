# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pinned official PP-OCRv3 BGR preprocessing; fixed 640/48x320 deployment."""
import math
import cv2
import numpy as np

def prepare(bgr, role):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError('Expected a decoded uint8 BGR image')
    if role == 'det':
        value = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        value = (value - np.array([.485,.456,.406],np.float32)) / np.array([.229,.224,.225],np.float32)
        return np.ascontiguousarray(value.transpose(2,0,1)[None])
    if role != 'rec':
        raise ValueError(role)
    resized_w = min(320, int(math.ceil(48 * bgr.shape[1] / float(bgr.shape[0]))))
    value = cv2.resize(bgr, (resized_w,48)).astype(np.float32).transpose(2,0,1) / 255
    value -= .5
    value /= .5
    result = np.zeros((1,3,48,320),np.float32)
    result[0,:,:,:resized_w] = value
    return result

