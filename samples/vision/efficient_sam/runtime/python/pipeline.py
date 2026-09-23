# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin EfficientSAM adapter over the shared SAM pipeline."""

from samples._shared.sam_stages import SAMPipeline


class EfficientSAMPipeline(SAMPipeline):
    """EfficientSAM pipeline with a fixed, image-only public prediction API."""

    def __init__(self, runner, binding):
        """Bind only an ``efficient_sam`` model pair.

        Raises:
            ValueError: if the supplied model binding belongs to another
                sample, before any runtime stage is constructed.
        """
        if getattr(getattr(binding, "selection", None), "sample", None) != "efficient_sam":
            raise ValueError("EfficientSAMPipeline requires an efficient_sam model binding")
        super().__init__(runner, binding)

    def predict(self, image):
        """Return the shared mask result for one image using the fixed prompt."""
        return super().predict(image)


__all__ = ["EfficientSAMPipeline"]
