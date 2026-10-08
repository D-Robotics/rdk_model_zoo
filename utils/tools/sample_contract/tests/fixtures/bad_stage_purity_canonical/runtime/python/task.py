"""Negative fixture: canonical stage names crossing purity boundaries.

The 2026-10-05 readable-runtime rollout made ``preprocess`` / ``infer`` /
``postprocess`` the primary stage spellings (``pre_process`` / ``forward`` /
``post_process`` remain compatibility aliases), so the checker must flag
violations under the canonical names and the canonical ``infer_`` prefix too.
``save_report`` stays a non-stage helper and must not be flagged.
"""

import urllib.request

import cv2

import numpy as np


class CanonicalTask:
    """Task whose canonical stages download, save files and write archives."""

    def preprocess(self, image):
        urllib.request.urlretrieve("http://example.com/m.bin", "m.bin")
        return image

    def infer(self, tensors):
        cv2.imwrite("debug.png", tensors)
        return tensors

    def postprocess(self, outputs):
        np.save("out.npy", outputs)
        return outputs

    def predict(self, source):
        return self.postprocess(self.infer(self.preprocess(source)))

    def infer_calibration(self, tensors):
        with open("calib.bin", "wb") as handle:
            handle.write(tensors)
        return tensors


def save_report(path, text):
    """Non-stage helper: writing here is legitimate and must stay clean."""

    with open(path, "w") as handle:
        handle.write(text)
