"""CLI-boundary fixture for the canonical stage-purity rules.

Module-level helpers in a sample-local ``cli.py`` are the application
boundary: ``run_prepare`` legitimately downloads when the user asks for
model preparation, so it must surface as a recorded skip, never as a model
stage finding.  Stage-named *methods* inside this file (should any appear)
are still model logic and stay checked, as does every function in
non-CLI files.
"""

import urllib.request

import cv2


def run_prepare(args):
    """Module-level application helper: the one network-capable action."""

    urllib.request.urlretrieve(args.url, args.destination)
    return 0


class EmbeddedStage:
    """A stage method inside a CLI file is model logic, not an app helper."""

    def postprocess(self, outputs):
        cv2.imwrite("embedded.png", outputs)
        return outputs
