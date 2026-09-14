"""
LPRNet inference wrapper and decoding utilities.

This module implements the LPRNet recognition pipeline on RDK X5 with
`hbm_runtime`. The sample keeps the original runtime protocol used by the
legacy LPRNet demo:

- input tensor: packed `float32` binary data
- input shape: `1 x 3 x 24 x 94`
- output tensor: character logits with shape `68 x 18`

The wrapper is responsible for loading the model, reading the binary input
tensor, executing inference, and decoding the final plate string with a
CTC-style de-duplication rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import hbm_runtime
import numpy as np


CHARS: List[str] = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁",
    "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "I", "O", "-"
]


def decode_plate(pred_data: np.ndarray) -> str:
    """
    Decode the raw model output into the final license plate string.

    Args:
        pred_data (np.ndarray): Raw model output after squeezing the runtime
            tensor. The expected shape is `68 x 18`, where the first dimension
            is the character vocabulary and the second dimension is the time
            step.

    Returns:
        str: The decoded license plate string after removing repeated
        characters and blank labels.
    """

    pred_label = np.argmax(pred_data, axis=0)
    decoded = []
    previous = pred_label[0]
    if previous != len(CHARS) - 1:
        decoded.append(previous)

    for current in pred_label:
        if current == previous or current == len(CHARS) - 1:
            if current == len(CHARS) - 1:
                previous = current
            continue
        decoded.append(current)
        previous = current

    return "".join(CHARS[index] for index in decoded)


@dataclass
class LPRNetConfig:
    """
    Configuration for initializing the LPRNet runtime wrapper.

    Args:
        model_path (str): Path to the LPRNet `.bin` model.
        test_bin (str): Path to the packed `float32` binary input tensor.
    """

    model_path: str
    test_bin: str


class LPRNet:
    """
    LPRNet license plate recognition wrapper based on `hbm_runtime`.

    This wrapper encapsulates model loading, scheduling configuration, binary
    input loading, forward execution, and CTC-style output decoding for
    LPRNet recognition on RDK X5.
    """

    def __init__(self, config: LPRNetConfig):
        """
        Initialize the model and resolve runtime metadata.

        Args:
            config (LPRNetConfig): Runtime configuration containing the model
                path and the binary input tensor path.

        The initialization stage loads the `.bin` model and caches the input
        and output tensor metadata required by the later inference stages.
        """

        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(self.cfg.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]

    def set_scheduling_params(self, priority: Optional[int] = None, bpu_cores: Optional[List[int]] = None) -> None:
        """
        Configure runtime scheduling parameters.

        Args:
            priority (Optional[int]): Runtime priority passed to
                `HB_HBMRuntime.set_scheduling_params`.
            bpu_cores (Optional[List[int]]): BPU core indexes used for
                inference execution.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load the packed `float32` tensor used as the LPRNet input.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Nested input dictionary in the
            format required by `HB_HBMRuntime.run`.

        The sample uses a pre-packed `float32` tensor instead of an image
        preprocessing pipeline. The tensor is reshaped according to the model
        input metadata before inference.
        """

        input_shape = self.input_shapes[self.input_names[0]]
        input_data = np.fromfile(self.cfg.test_bin, dtype=np.float32).reshape(input_shape)
        return {self.model_name: {self.input_names[0]: input_data}}

    def forward(self, input_tensors: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Run BPU inference and return the raw logits tensor.

        Args:
            input_tensors (Dict[str, Dict[str, np.ndarray]]): Runtime input
                dictionary produced by `pre_process`.

        Returns:
            np.ndarray: The squeezed logits tensor returned by the runtime.
        """

        outputs = self.model.run(input_tensors)
        return outputs[self.model_name][self.output_names[0]].squeeze()

    def post_process(self, logits: np.ndarray) -> str:
        """
        Decode the logits tensor into the final plate string.

        Args:
            logits (np.ndarray): Raw logits tensor returned by the runtime.

        Returns:
            str: Final decoded license plate string.
        """

        return decode_plate(logits)

    def predict(self) -> str:
        """
        Run the complete LPRNet pipeline on one binary input tensor.

        Returns:
            str: Final decoded license plate string produced by the model.
        """

        input_tensors = self.pre_process()
        outputs = self.forward(input_tensors)
        return self.post_process(outputs)

    def __call__(self) -> str:
        """
        Provide functional-style inference capability.

        Returns:
            str: Same result as `predict()`.
        """

        return self.predict()
