# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three planning stages: logical features, raw inference and decoded predictions.

``predict`` composes ``preprocess`` → ``infer`` → ``postprocess``; the
established ``pre_process``/``forward``/``post_process`` names stay thin
aliases of those implementations.
"""

from dataclasses import dataclass

import numpy as np
from samples.vision.diffusiondrive.runtime.python.quantization import quantize, decode


@dataclass(frozen=True)
class DiffusionDriveDetails:
    """One predict call's decoded result plus its physical inputs and raw outputs.

    Callers that archive ``physical_inputs.npz``/``raw_outputs.npz`` request
    this record with ``return_details=True`` instead of recomputing stages.
    It describes only its own call; the task never retains a last output.
    """

    result: dict
    physical: dict
    raw: dict


class DiffusionDriveTask:
    def __init__(self, runner, binding, agent_score_threshold=0.5):
        if (
            not np.isfinite(agent_score_threshold)
            or not 0 <= agent_score_threshold <= 1
        ):
            raise ValueError("Agent score threshold must be finite and within [0,1]")
        self.runner = runner
        self.binding = binding
        self.agent_score_threshold = float(agent_score_threshold)

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, features):
        """Exact four float32 feature arrays to metadata-declared physical IO."""
        if set(features) != set(self.binding.input_transforms):
            raise ValueError("Exact four logical input names required")
        return {
            name: quantize(features[name], spec)
            for name, spec in self.binding.input_transforms.items()
        }

    def infer(self, prepared):
        """Return all raw named outputs; never dequantize or render here."""
        return self.runner(prepared)

    def postprocess(self, outputs):
        """Raw physical tensors to owned trajectory/agents/BEV, no filesystem IO."""
        if set(outputs) != set(self.binding.output_transforms):
            raise ValueError("Exact four raw output names required")
        decoded = {
            name: decode(outputs[name], spec)
            for name, spec in self.binding.output_transforms.items()
        }
        scores = 1 / (1 + np.exp(-np.clip(decoded["agent_labels"], -60, 60)))
        return {
            "trajectory": decoded["trajectory"],
            "agent_states": decoded["agent_states"],
            "agent_scores": scores,
            "agent_mask": scores >= self.agent_score_threshold,
            "bev_logits": decoded["bev_semantic_map"],
            "bev_labels": np.argmax(decoded["bev_semantic_map"], axis=1).astype(
                np.uint8
            ),
        }

    def predict(self, features, *, return_details=False):
        """Compose the same three stages once with fixed caller-provided noise.

        ``return_details=True`` wraps the usual decoded mapping with this
        call's physical inputs and raw outputs, so one production inference
        also serves callers that archive the raw IO; the default return stays
        the decoded mapping alone.
        """
        physical = self.preprocess(features)
        raw = self.infer(physical)
        result = self.postprocess(raw)
        if return_details:
            return DiffusionDriveDetails(result, physical, raw)
        return result

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, features):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(features)

    def forward(self, prepared):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(prepared)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)
