"""Fixture task module with a contract-clean three-stage interface."""


def _resize(image, size):
    """Pure helper: pretend resize without any external library."""

    return (image, size)


class FixtureTask:
    """Single-stage fixture task (pre/forward/post/predict)."""

    def __init__(self, selection):
        self.selection = selection
        self.runner = None

    def pre_process(self, image):
        """Return prepared tensors for one BGR uint8 image."""

        return _resize(image, (224, 224))

    def forward(self, tensors):
        """Feed the bound tensor contract; no decode, no file access."""

        if self.runner is None:
            raise RuntimeError("runner not loaded")
        return self.runner.run(tensors)

    def post_process(self, outputs):
        """Reduce raw outputs to the top-5 (label, score) list."""

        scores = sorted(outputs, reverse=True)
        return scores[:5]

    def predict(self, image):
        """Chain the three stages and return the Result contract."""

        return self.post_process(self.forward(self.pre_process(image)))

    def __call__(self, image):
        """Delegate to predict."""

        return self.predict(image)
