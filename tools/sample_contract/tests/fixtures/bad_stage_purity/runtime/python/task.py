"""Negative fixture: stage functions that violate the inference contract.

Each violation is AST-detectable: a download, a file save, a write-mode
open, and a subprocess call.  ``save_report`` is deliberately NOT a stage
function and must not be flagged (scope precision).
"""

import subprocess
import urllib.request


class BadTask:
    """Task whose stages cross the download/save/subprocess boundaries."""

    def pre_process(self, image):
        with open("cache.txt", "w") as handle:
            handle.write("stale context")
        return image

    def forward(self, tensors):
        urllib.request.urlretrieve("http://example.com/m.bin", "m.bin")
        import cv2

        cv2.imwrite("debug.png", tensors)
        return tensors

    def post_process(self, outputs):
        subprocess.run(["echo", "post"])
        return outputs

    def predict(self, image):
        return self.post_process(self.forward(self.pre_process(image)))


def save_report(path, text):
    """Non-stage helper: writing here is legitimate and must stay clean."""

    with open(path, "w") as handle:
        handle.write(text)
