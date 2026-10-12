"""SDK and environment doubles shared by the LiteAvatar host tests.

``FakeRuntime`` mimics the public surface ``hbm_runtime.HB_HBMRuntime``
exposes on the board (nested per-model metadata, ``run`` over one model),
with the boundary the four golden S100 artifacts were verified to have:
FLOAT32 tensors and no output quantization.  It never imports a board SDK.

``ArrayFrontend`` and ``numpy_interpolate`` replace the two Torch-dependent
steps (fbank, time interpolation) with deterministic NumPy arithmetic so the
full pipeline predict flow runs on a Torch-free host; the real numerics are
pinned by the server parity run and the board comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from samples.vision.lite_avatar.runtime.python.avatar import INPUTS, OUTPUTS
from samples.vision.lite_avatar.runtime.python import frontend


def stage_contracts(stage):
    """(input names, shapes, dtypes, output names, shapes, dtypes) for stage."""

    inputs = INPUTS[stage]
    outputs = OUTPUTS[stage]
    in_names = tuple(next(iter(aliases)) for aliases, _, _ in inputs.values())
    out_names = tuple(next(iter(aliases)) for aliases, _, _ in outputs.values())
    in_shapes = {n: c[1] for n, c in zip(in_names, inputs.values())}
    in_dtypes = {n: c[2] for n, c in zip(in_names, inputs.values())}
    out_shapes = {n: c[1] for n, c in zip(out_names, outputs.values())}
    out_dtypes = {n: c[2] for n, c in zip(out_names, outputs.values())}
    return in_names, in_shapes, in_dtypes, out_names, out_shapes, out_dtypes


class FakeRuntime:
    """One-model board runtime double with F32 boundary and deterministic run."""

    def __init__(self, stage, *, dtype_token="F32", transform=None):
        (
            self.input_names_l,
            self.input_shapes_l,
            self.input_dtypes_l,
            self.output_names_l,
            self.output_shapes_l,
            self.output_dtypes_l,
        ) = stage_contracts(stage)
        self.stage = stage
        self.model_names = [stage]
        self.calls = []
        self.scheduling_requests = []
        self.transform = transform
        # Board evidence 2026-10-12: the four golden artifacts report nested
        # per-model metadata and F32/NONE quantization at this boundary.
        self.input_names = {stage: list(self.input_names_l)}
        self.input_shapes = {stage: dict(self.input_shapes_l)}
        self.input_dtypes = {
            stage: {name: dtype_token for name in self.input_names_l}
        }
        self.output_names = {stage: list(self.output_names_l)}
        self.output_shapes = {stage: dict(self.output_shapes_l)}
        self.output_dtypes = {
            stage: {name: dtype_token for name in self.output_names_l}
        }
        self.output_quants = {stage: {name: None for name in self.output_names_l}}

    def _outputs(self, feed):
        if self.transform is not None:
            return self.transform(feed)
        return {
            name: np.zeros(shape, np.float32)
            for name, shape in self.output_shapes_l.items()
        }

    def run(self, feed):
        model = self.model_names[0]
        self.calls.append(feed)
        return {model: self._outputs(feed[model])}

    def set_scheduling_params(self, **kwargs):
        self.scheduling_requests.append(kwargs)

    def __call__(self, tensors):
        """NamedArrayRunner-shaped call: owned dict of the model's outputs."""

        result = self.run({self.model_names[0]: dict(tensors)})
        return {
            name: np.array(value, copy=True)
            for name, value in result[self.model_names[0]].items()
        }


class ArrayFrontend(frontend.AvatarFrontend):
    """Deterministic Torch-free stand-in for the fbank stage.

    Produces the same geometry the real frontend would for ``len(audio)``
    16 kHz samples: one LFR frame per ~60 ms of audio, with a per-frame
    ramp in coefficient 0 so downstream content is traceable.
    """

    def features_from_mono(self, audio):
        frames = max((len(audio) - 400) // 160 + 1, 1)
        mat = np.zeros((frames, frontend.N_MELS), np.float32)
        mat[:, 0] = np.linspace(0.0, 1.0, frames, dtype=np.float32)
        stacked = frontend.apply_lfr(mat)
        return frontend.apply_cmvn(stacked, self.cmvn)


def numpy_interpolate(hidden_states, frame_count):
    """Torch-free test double for ``avatar.interpolate_time_axis``.

    Linear resampling of the time axis with endpoint alignment, matching
    the structure (not the bitwise numerics) of the Torch implementation.
    """

    positions = np.linspace(0.0, hidden_states.shape[0] - 1, frame_count)
    low = np.floor(positions).astype(int)
    high = np.ceil(positions).astype(int)
    weight = (positions - low)[:, None, None].astype(np.float32)
    blended = hidden_states[low] * (1 - weight) + hidden_states[high] * weight
    return np.ascontiguousarray(blended, dtype=np.float32)


def build_avatar_data(directory, frames=6, width=64, height=48, refs=None, box=None):
    """Write a minimal synthetic avatar data directory (cv2, real files)."""

    import cv2

    directory = Path(directory)
    (directory / "ref_frames").mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(directory / "bg_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("cv2 could not open an mp4 writer for the test data")
    for index in range(frames):
        tile = np.full((height, width, 3), 40 + index * 8, np.uint8)
        tile[:, : width // 4] = 200
        writer.write(tile)
    writer.release()
    y1, y2, x1, x2 = box or (10, height - 10, 12, width - 12)
    (directory / "face_box.txt").write_text(f"{y1} {y2} {x1} {x2}\n", encoding="utf-8")
    for index in range(frames if refs is None else refs):
        reference = np.full((height, width, 3), 30 + index * 5, np.uint8)
        cv2.imwrite(str(directory / "ref_frames" / f"ref_{index:05d}.jpg"), reference)
    return directory
