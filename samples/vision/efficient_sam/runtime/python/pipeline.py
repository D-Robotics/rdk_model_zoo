# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""The readable EfficientSAM encoder → decoder pipeline.

This file shows the sample's real two-model composition: the encoder turns
one image into an image embedding, the decoder turns that embedding into
mask candidates, and :meth:`EfficientSAMPipeline.predict` chains the two
stages with per-stage error attribution.  The SAM math itself stays in the
shared implementations — tensor preparation in ``samples/_shared/sam_tensor_io.py``
and metadata validation in ``samples/_shared/sam_binding.py`` — so nothing
here reimplements normalization, box validation or mask resizing.

The decoder of the published EfficientSAM artifacts uses the prompt that
was fixed at export time; a runtime box is rejected (the shared decoder
preparation raises for it).
"""

from __future__ import annotations

from typing import Any

from samples._shared.sam_stages import DecoderStage, EncoderStage, SAMPipeline, StageError


class EfficientSAMEncoder(EncoderStage):
    """Encoder stage with the canonical three-step spelling.

    ``preprocess``/``infer``/``postprocess`` are thin canonical aliases of
    the inherited shared implementation (``pre_process``/``forward``/
    ``post_process``); there is exactly one implementation.
    """

    def preprocess(self, image: Any):
        """Prepare one BGR image as the bound float32 RGB NCHW tensors."""
        return self.pre_process(image)

    def infer(self, prepared):
        """Execute exactly one encoder call on the prepared tensors."""
        return self.forward(prepared)

    def postprocess(self, outputs):
        """Own the validated float32 (1,256,32,32) image embedding."""
        return self.post_process(outputs)


class EfficientSAMDecoder(DecoderStage):
    """Decoder stage with the canonical three-step spelling."""

    def preprocess(self, embedding: Any, *, box=None):
        """Prepare the embedding (and, when supported, a box prompt)."""
        return self.pre_process(embedding, box=box)

    def infer(self, prepared):
        """Execute exactly one decoder call on the prepared tensors."""
        return self.forward(prepared)

    def postprocess(self, outputs):
        """Select the best-IoU candidate and resize it to a 512-square mask."""
        return self.post_process(outputs)


class EfficientSAMPipeline(SAMPipeline):
    """EfficientSAM pipeline with the real orchestration readable here.

    The two stage objects stay public (``encoder``/``decoder``) with their
    three-step APIs; ``encode_image``/``decode_masks`` run one stage each,
    and ``predict`` shows the composition.  Errors are attributed with the
    shared :class:`StageError` so callers can tell encoder from decoder
    failures; the failing stage's successor is never executed.
    """

    def __init__(self, runner, binding):
        """Bind only an ``efficient_sam`` model pair.

        Raises:
            ValueError: if the supplied model binding belongs to another
                sample, before any runtime stage is constructed.
        """
        if getattr(getattr(binding, "selection", None), "sample", None) != "efficient_sam":
            raise ValueError("EfficientSAMPipeline requires an efficient_sam model binding")
        super().__init__(runner, binding)
        # Canonical-name views over the same shared stage implementations.
        self.encoder = EfficientSAMEncoder(runner.encoder, binding.encoder)
        self.decoder = EfficientSAMDecoder(runner.decoder, binding.decoder)

    def encode_image(self, image):
        """Run the encoder's three public steps and return the embedding."""
        prepared = self.encoder.preprocess(image)
        raw = self.encoder.infer(prepared)
        return self.encoder.postprocess(raw)

    def decode_masks(self, embedding):
        """Run the decoder's three public steps for one embedding."""
        prepared = self.decoder.preprocess(embedding)
        raw = self.decoder.infer(prepared)
        return self.decoder.postprocess(raw)

    def predict(self, image):
        """Encode one image, then decode the fixed-prompt mask result."""
        try:
            embedding = self.encode_image(image)
        except StageError:
            raise
        except Exception as exc:
            raise StageError("encoder", str(exc)) from exc
        try:
            return self.decode_masks(embedding)
        except StageError:
            raise
        except Exception as exc:
            raise StageError("decoder", str(exc)) from exc


__all__ = ["EfficientSAMDecoder", "EfficientSAMEncoder", "EfficientSAMPipeline"]
