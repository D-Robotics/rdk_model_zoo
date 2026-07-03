# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PP-LiteSeg-STDC1 model wrapper for RDK X5 BPU inference."""

import cv2
import numpy as np

try:
    from hbm_runtime import HB_HBMRuntime
except ImportError:
    raise ImportError("hbm_runtime not found — run on RDK X5 with firmware >= 3.5.0")


# Cityscapes 19-class palette (BGR for OpenCV)
CITYSCAPES_PALETTE_BGR = np.array([
    [128,  64, 128],  # 0  road
    [232,  35, 244],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [156, 102, 102],  # 3  wall
    [153, 153, 190],  # 4  fence
    [153, 153, 153],  # 5  pole
    [ 30, 170, 250],  # 6  traffic light
    [  0, 220, 220],  # 7  traffic sign
    [ 35, 142, 107],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [180, 130,  70],  # 10 sky
    [ 60,  20, 220],  # 11 person
    [  0,   0, 255],  # 12 rider
    [142,   0,   0],  # 13 car
    [100,  60,   0],  # 14 truck
    [ 70,   0,   0],  # 15 bus
    [100,  80,   0],  # 16 train
    [230,   0,   0],  # 17 motorcycle
    [ 32,  11, 119],  # 18 bicycle
], dtype=np.uint8)

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


class PPLiteSegConfig:
    """Configuration for PP-LiteSeg-STDC1 inference."""

    def __init__(
        self,
        model_path: str,
        input_width: int = 1024,
        input_height: int = 512,
        alpha: float = 0.55,
    ):
        self.model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.alpha = alpha


class PPLiteSeg:
    """PP-LiteSeg-STDC1 BPU inference wrapper.

    Encapsulates the full inference pipeline:
        pre_process -> forward -> post_process -> predict
    """

    def __init__(self, config: PPLiteSegConfig):
        self.config = config
        self._runtime = HB_HBMRuntime(config.model_path)
        self._mname = self._runtime.model_names[0]
        self._iname = self._runtime.input_names[self._mname][0]
        self._oname = self._runtime.output_names[self._mname][0]

    @property
    def model(self):
        return self._runtime

    def set_scheduling_params(self, priority: int = 0, bpu_cores: list = None):
        """Configure BPU scheduling (no-op if runtime does not support it)."""
        pass

    def pre_process(self, bgr: np.ndarray) -> np.ndarray:
        """Convert BGR image to NV12 tensor for BPU input.

        Args:
            bgr: Input image as (H, W, 3) uint8 BGR array.

        Returns:
            NV12 array of shape (input_height * 3 // 2, input_width) uint8.
        """
        w, h = self.config.input_width, self.config.input_height
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        yuv_i420 = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV_I420)
        # I420 -> NV12: interleave U and V planes
        y = yuv_i420[:h, :]
        u = yuv_i420[h: h + h // 4].reshape(h // 2, w // 2)
        v = yuv_i420[h + h // 4:].reshape(h // 2, w // 2)
        uv = np.stack([u, v], axis=-1).reshape(h // 2, w)
        nv12 = np.vstack([y, uv])
        return np.ascontiguousarray(nv12, dtype=np.uint8)

    def forward(self, nv12: np.ndarray) -> np.ndarray:
        """Run BPU inference.

        Args:
            nv12: Preprocessed NV12 array.

        Returns:
            Raw output array of shape (1, H, W, 1) int32.
        """
        results = self._runtime.run({self._iname: nv12})
        return results[self._mname][self._oname]

    def post_process(self, raw: np.ndarray) -> np.ndarray:
        """Squeeze raw output to a 2-D segmentation map.

        Args:
            raw: Raw BPU output of shape (1, H, W, 1) int32.

        Returns:
            Segmentation map of shape (H, W) int32 with class indices.
        """
        return raw.squeeze().astype(np.int32)

    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """End-to-end inference: BGR image -> segmentation map.

        Args:
            bgr: Input image as (H, W, 3) uint8 BGR array.

        Returns:
            Segmentation map of shape (input_height, input_width) int32.
        """
        nv12 = self.pre_process(bgr)
        raw = self.forward(nv12)
        return self.post_process(raw)

    def colorize(self, seg: np.ndarray) -> np.ndarray:
        """Map class indices to BGR colors using the Cityscapes palette."""
        h, w = seg.shape
        out = np.zeros((h, w, 3), dtype=np.uint8)
        for cid in range(len(CITYSCAPES_PALETTE_BGR)):
            out[seg == cid] = CITYSCAPES_PALETTE_BGR[cid]
        return out

    def draw_legend(self, canvas: np.ndarray, cls_ids: list) -> np.ndarray:
        """Overlay a small class legend on the top-right corner of canvas."""
        box, pad = 18, 6
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
        leg_h = (box + pad) * len(cls_ids) + pad
        leg_w = 155
        leg = np.full((leg_h, leg_w, 3), 30, dtype=np.uint8)
        for i, cid in enumerate(cls_ids):
            y0 = pad + i * (box + pad)
            c = [int(x) for x in CITYSCAPES_PALETTE_BGR[cid]]
            cv2.rectangle(leg, (pad, y0), (pad + box, y0 + box), c, -1)
            cv2.putText(leg, CITYSCAPES_CLASS_NAMES[cid], (pad + box + 4, y0 + box - 3),
                        font, fs, (220, 220, 220), th)
        h, w = canvas.shape[:2]
        canvas[4: 4 + leg_h, w - leg_w - 4: w - 4] = leg
        return canvas

    def visualize(self, bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """Produce a 3-panel result image: Original | Overlay | Segmentation.

        Args:
            bgr: Original BGR input image (any size).
            seg: Segmentation map (input_height, input_width) int32.

        Returns:
            Concatenated result image (H+header, 3*W+dividers, 3) uint8.
        """
        w, h = self.config.input_width, self.config.input_height
        alpha = self.config.alpha

        seg_color = self.colorize(seg)
        orig_rsz = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(orig_rsz, 1 - alpha, seg_color, alpha, 0)

        unique = sorted(np.unique(seg).tolist())
        valid = [c for c in unique if 0 <= c < len(CITYSCAPES_CLASS_NAMES)]
        overlay = self.draw_legend(overlay, valid)

        div = np.full((h, 3, 3), 60, dtype=np.uint8)
        panel = np.hstack([orig_rsz, div, overlay, div, seg_color])

        hdr = np.full((36, panel.shape[1], 3), 35, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(hdr, "Original", (10, 25), font, 0.65, (200, 200, 200), 1)
        cv2.putText(hdr, f"Overlay  alpha={alpha:.2f}", (w + 13, 25), font, 0.65, (200, 200, 200), 1)
        cv2.putText(hdr, "Segmentation", (2 * w + 16, 25), font, 0.65, (200, 200, 200), 1)
        return np.vstack([hdr, panel])
