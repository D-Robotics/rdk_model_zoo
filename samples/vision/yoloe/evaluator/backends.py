# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU ONNX and board backends with shared task postprocessing."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import numpy as np
from utils.py_utils.assets import sha256_file
from utils.py_utils.yoloe26_geometry import letterbox
from samples.vision.ultralytics_yolo.runtime.python.geometry import (
    resize_with_transform,
)
from samples.vision.yoloe.conversion.contract import inspect_graph
from samples.vision.yoloe.runtime.python.model_binding import (
    resolve_selection,
    runtime_selection,
)
from samples.vision.yoloe.runtime.python.config import validate_config
from samples.vision.yoloe.runtime.python.yoloe import decode_result
from samples.vision.yoloe.runtime.python.yoloe import YOLOE

NAMES = Path(__file__).resolve().parents[1] / "test_data/classes.names"


@dataclass(frozen=True)
class RGBPrepared:
    tensor: np.ndarray
    context: object


class OnnxPredictor:
    """Float RGB evaluation; it does not simulate NV12 conversion or a BPU."""

    def __init__(self, selection, cfg, model_path, digest, threads=2):
        import onnxruntime as ort

        validate_config(selection, cfg)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be positive.")
        self.selection, self.cfg = selection, cfg
        self.contract = runtime_selection(selection).contract
        facts = inspect_graph(model_path, NAMES, selection.variant)
        if facts["onnx_sha256"] != digest:
            raise ValueError("ONNX SHA-256 mismatch.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        if sha256_file(model_path) != digest:
            raise ValueError("ONNX changed while loading.")
        self.roles = facts["output_roles"]
        self.identity = {
            "backend": "onnx",
            "model_path": str(model_path),
            "model_sha256": digest,
            "target_protocol": selection.target,
            "variant": selection.variant,
            "source_asset_id": selection.asset.reference,
            "input": "RGB float32 NCHW 0..1; no NV12 roundtrip",
            "onnxruntime": ort.__version__,
            "graph_optimization": "ORT_DISABLE_ALL",
            "threads": threads,
            "config": asdict(cfg),
            "board": "not-run",
        }

    def pre_process(self, image):
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or min(image.shape[:2]) <= 0
        ):
            raise ValueError("Expected nonempty BGR uint8 HWC image.")
        pixels, context = (
            letterbox(image)
            if self.selection.variant.startswith("26")
            else resize_with_transform(image, (640, 640), self.cfg.resize_type)
        )
        tensor = (
            np.ascontiguousarray(
                pixels[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32
            )
            / 255
        )
        return RGBPrepared(tensor, context)

    def forward(self, prepared):
        names = list(self.roles.values())
        outputs = self.session.run(names, {"images": prepared.tensor})
        return dict(zip(self.roles, outputs, strict=True))

    def post_process(self, raw, context):
        return decode_result(raw, self.contract, self.selection, self.cfg, context)

    def predict(self, image):
        prepared = self.pre_process(image)
        return self.post_process(self.forward(prepared), prepared.context)


class BoardPredictor:
    def __init__(self, selection, cfg):
        self.task = YOLOE(selection, cfg)
        self.identity = {
            "backend": "board",
            "model_path": str(selection.model_path),
            "model_sha256": selection.local_float_sha256,
            "target": selection.target,
            "variant": selection.variant,
            "source_asset_id": selection.asset.reference,
            "input": "NV12 through installed board SDK",
            "config": asdict(cfg),
        }

    def predict(self, image):
        return self.task.predict(image)


def create_predictor(
    backend, target, variant, model_path, model_sha256, cfg, threads=2
):
    if (
        not isinstance(model_sha256, str)
        or re.fullmatch("[0-9a-fA-F]{64}", model_sha256) is None
    ):
        raise ValueError(
            "Provide the exact 64-digit SHA-256 of the evaluated model file."
        )
    path = Path(model_path).expanduser().resolve()
    digest = model_sha256.lower()
    if sha256_file(path) != digest:
        raise ValueError("Model SHA-256 mismatch.")
    selection = resolve_selection(target, variant=variant)
    if backend == "onnx":
        return OnnxPredictor(selection, cfg, path, digest, threads)
    if backend == "board":
        if threads != 2:
            raise ValueError(
                "--threads configures ONNX only; it cannot change board scheduling."
            )
        local = resolve_selection(
            target, variant=variant, model_path=path, local_float_sha256=digest
        )
        return BoardPredictor(local, cfg)
    raise ValueError(f"Unknown evaluation backend: {backend}")
