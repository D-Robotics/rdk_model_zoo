# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""The readable MobileSAM encoder → decoder pipeline with box prompts.

This file shows the sample's real two-model composition: the encoder turns
one image into an image embedding, the decoder turns that embedding plus
one box prompt into mask candidates, and
:meth:`MobileSAMPipeline.predict` chains the two stages with per-stage
error attribution.  The SAM math itself stays in the shared
implementations — tensor preparation (including the box validation and the
per-target box shape) in ``utils/py_utils/sam_tensor_io.py`` and metadata
validation in ``utils/py_utils/sam_binding.py`` — so nothing here
reimplements normalization, prompt encoding or mask resizing.

The published MobileSAM decoder accepts one box prompt per call in resized
512-image coordinates; the established default is the shared
``DEFAULT_BOX``.
"""

from __future__ import annotations

from typing import Any

from utils.py_utils.sam_stages import DecoderStage, EncoderStage, SAMPipeline, StageError
from utils.py_utils.sam_tensor_io import DEFAULT_BOX


class MobileSAMEncoder(EncoderStage):
    """Encoder stage with the canonical three-step spelling.

    ``preprocess``/``infer``/``postprocess`` are thin canonical aliases of
    the inherited shared implementation (``pre_process``/``forward``/
    ``post_process``); there is exactly one implementation.
    """

    def preprocess(self, image: Any):
        """Prepare one BGR image as the bound normalized float32 RGB NCHW tensors."""
        return self.pre_process(image)

    def infer(self, prepared):
        """Execute exactly one encoder call on the prepared tensors."""
        return self.forward(prepared)

    def postprocess(self, outputs):
        """Own the validated float32 (1,256,32,32) image embedding."""
        return self.post_process(outputs)


class MobileSAMDecoder(DecoderStage):
    """Decoder stage with the canonical three-step spelling and box prompt."""

    def preprocess(self, embedding: Any, *, box=None):
        """Prepare the embedding and the box prompt for one decode call."""
        return self.pre_process(embedding, box=box)

    def infer(self, prepared):
        """Execute exactly one decoder call on the prepared tensors."""
        return self.forward(prepared)

    def postprocess(self, outputs):
        """Select the best-IoU candidate and resize it to a 512-square mask."""
        return self.post_process(outputs)


class MobileSAMPipeline(SAMPipeline):
    """MobileSAM pipeline with the real orchestration readable here.

    The two stage objects stay public (``encoder``/``decoder``) with their
    three-step APIs; ``encode_image``/``decode_masks`` run one stage each,
    and ``predict`` shows the composition, forwarding the per-call box
    prompt.  Errors are attributed with the shared :class:`StageError` so
    callers can tell encoder from decoder failures; the failing stage's
    successor is never executed.

    Attributes:
        runner: Owner of the loaded encoder and decoder SDK runtimes.
        binding: Validated model-pair metadata and artifact selection.
        encoder: Public image-embedding stage.
        decoder: Public mask-decoding stage.
    """

    def __init__(self, runner, binding):
        """Bind only a ``mobile_sam`` model pair.

        Args:
            runner: Loaded encoder/decoder runtime pair.
            binding: Validated model-pair metadata for this sample.

        Raises:
            ValueError: if the supplied model binding belongs to another
                sample, before any runtime stage is constructed.
        """
        if getattr(getattr(binding, "selection", None), "sample", None) != "mobile_sam":
            raise ValueError("MobileSAMPipeline requires a mobile_sam model binding")
        self.runner = runner
        self.binding = binding
        super().__init__(runner, binding)
        # Canonical-name views over the same shared stage implementations.
        self.encoder = MobileSAMEncoder(runner.encoder, binding.encoder)
        self.decoder = MobileSAMDecoder(runner.decoder, binding.decoder)

    @classmethod
    def from_models(cls, selection, *, runtime_factory=None):
        """Load and bind the encoder/decoder pair and construct the pipeline.

        Args:
            selection: Resolved mobile_sam artifact paths and target from cli.
            runtime_factory: Optional SDK-compatible factory for injected runtimes.
                None uses the board SDK after checking the executing board.

        Returns:
            MobileSAMPipeline: Pipeline owning both loaded models and metadata.

        Raises:
            ValueError: The selection belongs to another sample or board.
            RuntimeError: SDK loading or model initialization fails.
            MetadataMismatchError: Encoder/decoder tensors violate the contract.
        """
        if getattr(selection, "sample", None) != "mobile_sam":
            raise ValueError("MobileSAMPipeline requires a mobile_sam model selection")
        from utils.py_utils.sam_runner import RuntimeModelRunner

        runner = RuntimeModelRunner(selection, runtime_factory=runtime_factory)
        binding = runner.load()
        return cls(runner, binding)

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Apply scheduling settings to the encoder and decoder models.

        Args:
            priority: Native scheduling priority in 0..255, or None to retain it.
            bpu_cores: Nonempty list of nonnegative S-series core indexes.
                Use None on X5, which does not expose core selection here.

        Returns:
            None.

        Raises:
            ValueError: The scheduling arguments are invalid for the target.
            RuntimeError: Either SDK stage rejects the scheduling request.
        """
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

    def encode_image(self, image):
        """Run the encoder's three public steps and return the embedding."""
        prepared = self.encoder.preprocess(image)
        raw = self.encoder.infer(prepared)
        return self.encoder.postprocess(raw)

    def decode_masks(self, embedding, *, box=None):
        """Run the decoder's three public steps for one embedding and box."""
        prepared = self.decoder.preprocess(embedding, box=box)
        raw = self.decoder.infer(prepared)
        return self.decoder.postprocess(raw)

    def predict(self, image, *, box=DEFAULT_BOX):
        """Encode one image, then decode the box-prompted mask result."""
        try:
            embedding = self.encode_image(image)
        except StageError:
            raise
        except Exception as exc:
            raise StageError("encoder", str(exc)) from exc
        try:
            return self.decode_masks(embedding, box=box)
        except StageError:
            raise
        except Exception as exc:
            raise StageError("decoder", str(exc)) from exc


__all__ = ["MobileSAMDecoder", "MobileSAMEncoder", "MobileSAMPipeline"]
