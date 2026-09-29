"""One named, fixed-contract CPU ONNX call; no CIF, decoding or file preparation."""

import numpy as np
from samples.speech.paraformer.conversion.export import signature


class OnnxStage:
    def __init__(self, path, stage, threads=4):
        import onnxruntime as ort

        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be positive")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for side, tensors in (
            ("input", self.session.get_inputs()),
            ("output", self.session.get_outputs()),
        ):
            expected = signature(stage, side)
            if len(tensors) != len(expected):
                raise ValueError(f"Unexpected {stage} {side} count")
            for tensor, (name, shape, dtype) in zip(tensors, expected, strict=True):
                ort_type = "tensor(float)" if dtype == "float32" else "tensor(int32)"
                if (
                    tensor.name != name
                    or tuple(tensor.shape) != shape
                    or tensor.type != ort_type
                ):
                    raise ValueError(f"Unexpected {stage} {side} tensor {tensor.name}")
        self.inputs = {
            name: (shape, dtype) for name, shape, dtype in signature(stage, "input")
        }
        self.outputs = [name for name, _, _ in signature(stage, "output")]

    def forward(self, feed):
        if set(feed) != set(self.inputs):
            raise ValueError("ONNX feed must contain exactly the named stage inputs")
        for name, value in feed.items():
            shape, dtype = self.inputs[name]
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.dtype(dtype)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid ONNX input {name}")
        return {
            name: np.array(value, copy=True)
            for name, value in zip(
                self.outputs, self.session.run(self.outputs, feed), strict=True
            )
        }
