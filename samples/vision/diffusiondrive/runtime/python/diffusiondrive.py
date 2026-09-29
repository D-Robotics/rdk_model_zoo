# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three planning stages: logical features, raw inference and decoded predictions."""

import numpy as np
from samples.vision.diffusiondrive.runtime.python.quantization import quantize, decode


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

    def pre_process(self, features):
        """Exact four float32 feature arrays to metadata-declared physical IO."""
        if set(features) != set(self.binding.input_transforms):
            raise ValueError("Exact four logical input names required")
        return {
            name: quantize(features[name], spec)
            for name, spec in self.binding.input_transforms.items()
        }

    def forward(self, prepared):
        """Return all raw named outputs; never dequantize or render here."""
        return self.runner(prepared)

    def post_process(self, outputs):
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

    def predict(self, features):
        """Compose the same three stages once with fixed caller-provided noise."""
        return self.post_process(self.forward(self.pre_process(features)))
