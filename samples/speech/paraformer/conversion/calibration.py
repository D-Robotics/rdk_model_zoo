"""Representative WAV selection and shared unmasked CPU calibration bridge."""

from pathlib import Path
import numpy as np

from samples.speech.paraformer.runtime.python.cif import cif_numpy
from samples.speech.paraformer.runtime.python.cli import CONTEXT

CALIBRATION = {
    "speech": ((1, 400, 560), "float32"),
    "encoder_after_norm_Add_1_output_0": ((1, 400, 512), "float32"),
    "predictor_Add_output_0": ((1, 401), "float32"),
    "predictor_Concat_5_output_0": ((1, 401, 512), "float32"),
    "shape_8609": ((1, 100, 512), "float32"),
    "token_num": ((1,), "int32"),
    "bias_embed": ((1, 1, 512), "float32"),
}


def select_wavs(directory, count):
    """Choose a deterministic prefix; never silently accept an empty dataset."""
    directory = Path(directory).expanduser().resolve()
    if type(count) is not int or count < 1:
        raise ValueError("sample-count must be a positive integer")
    if not directory.is_dir():
        raise ValueError("wav-dir must be an existing directory")
    paths = tuple(sorted(path for path in directory.rglob("*.wav") if path.is_file()))
    if not paths:
        raise ValueError("No .wav files found")
    return paths[:count]


def checked_tensor(value, role):
    shape, dtype = CALIBRATION[role]
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype(dtype)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{role} requires finite {dtype} {shape}")
    return np.array(value, copy=True, order="C")


def intermediates(features, encoder, predictor):
    """Use real model outputs and unmasked CIF, preserving source calibration."""
    speech = checked_tensor(features, "speech")
    context = checked_tensor(
        encoder({"speech": speech.copy()})[CONTEXT], "encoder_after_norm_Add_1_output_0"
    )
    predicted = predictor({CONTEXT: context.copy()})
    alphas = checked_tensor(
        predicted["/predictor/Add_output_0"], "predictor_Add_output_0"
    )
    hidden = checked_tensor(
        predicted["/predictor/Concat_5_output_0"], "predictor_Concat_5_output_0"
    )
    acoustic, count = cif_numpy(alphas, hidden, real_T=None)
    return {
        "speech": speech,
        "encoder_after_norm_Add_1_output_0": context,
        "predictor_Add_output_0": alphas,
        "predictor_Concat_5_output_0": hidden,
        "shape_8609": checked_tensor(acoustic, "shape_8609"),
        "token_num": checked_tensor(count, "token_num"),
        "bias_embed": np.zeros((1, 1, 512), np.float32),
    }
