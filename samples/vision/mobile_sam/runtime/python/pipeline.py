# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin MobileSAM adapter over the shared SAM pipeline."""

from samples._shared.sam_stages import SAMPipeline


class MobileSAMPipeline(SAMPipeline):
    """MobileSAM pipeline with the shared numeric stages and box prompt."""

    def __init__(self, runner, binding):
        """Bind only a ``mobile_sam`` model pair.

        Raises:
            ValueError: if the supplied model binding belongs to another
                sample, before any runtime stage is constructed.
        """
        if getattr(getattr(binding, "selection", None), "sample", None) != "mobile_sam":
            raise ValueError("MobileSAMPipeline requires a mobile_sam model binding")
        super().__init__(runner, binding)

    def predict(self, image, *, box=(185.0, 120.0, 380.0, 445.0)):
        """Return the shared mask result for one image and a 512-space box."""
        return super().predict(image, box=box)


__all__ = ["MobileSAMPipeline"]
