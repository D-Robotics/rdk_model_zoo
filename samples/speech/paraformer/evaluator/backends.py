"""Named ONNX/HMCT execution adapters; inference math stays in the pipeline."""

from collections.abc import Mapping
import numpy as np

from samples.speech.paraformer.runtime.python.model_binding import bind_stage_io
from samples.speech.paraformer.runtime.python.pipeline import TensorNames


def graph_metadata(path, stage):
    import onnx

    graph = onnx.load(str(path), load_external_data=False)
    # Require self-contained inputs so the model digest identifies all weights.
    from onnx.external_data_helper import _get_all_tensors

    if any(
        t.data_location == onnx.TensorProto.EXTERNAL for t in _get_all_tensors(graph)
    ):
        raise ValueError("Evaluation requires self-contained ONNX models")
    initializers = {tensor.name for tensor in graph.graph.initializer}
    metadata = {"model_name": stage}
    for side, tensors in (("input", graph.graph.input), ("output", graph.graph.output)):
        tensors = [t for t in tensors if side != "input" or t.name not in initializers]
        metadata[f"{side}_names"] = tuple(t.name for t in tensors)
        metadata[f"{side}_shapes"] = {
            t.name: tuple(
                d.dim_value if d.HasField("dim_value") else None
                for d in t.type.tensor_type.shape.dim
            )
            for t in tensors
        }
        metadata[f"{side}_dtypes"] = {
            t.name: str(
                onnx.helper.tensor_dtype_to_np_dtype(t.type.tensor_type.elem_type)
            )
            for t in tensors
        }
    return metadata


class Stage:
    def __init__(self, path, stage, pipeline, threads=4):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be positive")
        if pipeline not in ("fp32", "int16"):
            raise ValueError("pipeline must be fp32 or int16")
        self.metadata = graph_metadata(path, stage)
        self.inputs, self.outputs = bind_stage_io(stage, self.metadata)
        self.pipeline = pipeline
        if pipeline == "fp32":
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        else:
            from hmct.executor import ORTExecutor

            self.session = ORTExecutor(str(path)).create_session()

    def _validate(self, values, side):
        names = self.metadata[f"{side}_names"]
        if not isinstance(values, Mapping) or set(values) != set(names):
            raise ValueError(f"Stage {side} names differ from declared ONNX interface")
        for name in names:
            value = values[name]
            if (
                not isinstance(value, np.ndarray)
                or value.shape != self.metadata[f"{side}_shapes"][name]
                or value.dtype != np.dtype(self.metadata[f"{side}_dtypes"][name])
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid stage {side}: {name}")

    def forward(self, feed):
        self._validate(feed, "input")
        names = self.metadata["output_names"]
        if self.pipeline == "fp32":
            outputs = dict(zip(names, self.session.run(list(names), feed), strict=True))
        else:
            outputs = self.session.forward(feed)
        self._validate(outputs, "output")
        return {name: np.array(value, copy=True) for name, value in outputs.items()}


def tensor_names(stages):
    enc, pred, dec = (stages[name] for name in ("encoder", "predictor", "decoder"))
    return TensorNames(
        enc.inputs["features"],
        enc.outputs["context"],
        pred.inputs["context"],
        pred.outputs["alphas"],
        pred.outputs["hidden"],
        dec.inputs["context"],
        dec.inputs["count"],
        dec.inputs["bias"],
        dec.inputs["acoustic"],
        dec.outputs["logits"],
    )
