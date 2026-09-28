"""Application-level three-model composition, with an explicit CPU CIF bridge.

Each injected callable performs one raw model execution with named tensors.
SDK loading, physical metadata binding, scheduling, audio/file I/O and reporting
belong to their respective adapters and application entry, not this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from numbers import Integral
from time import perf_counter

import numpy as np

from samples.speech.paraformer.runtime.python.cif import cif_numpy
from samples.speech.paraformer.runtime.python.decoding import (
    decode_logits,
    validate_vocabulary,
)


@dataclass(frozen=True)
class TensorNames:
    """Names supplied by model binding; no positional or substring fallback."""

    encoder_input: str
    encoder_output: str
    predictor_input: str
    predictor_alphas: str
    predictor_hidden: str
    decoder_context: str
    decoder_count: str
    decoder_bias: str
    decoder_acoustic: str
    decoder_logits: str

    def __post_init__(self):
        if any(
            not isinstance(getattr(self, field.name), str)
            or not getattr(self, field.name)
            for field in fields(self)
        ):
            raise ValueError("Every tensor role requires a nonempty name")
        if (
            self.predictor_alphas == self.predictor_hidden
            or len(
                {
                    self.decoder_context,
                    self.decoder_count,
                    self.decoder_bias,
                    self.decoder_acoustic,
                }
            )
            != 4
        ):
            raise ValueError("Distinct tensor roles cannot share a physical name")


@dataclass(frozen=True)
class Prediction:
    text: str
    token_ids: tuple[int, ...]
    token_count: int
    timings_ms: dict[str, float | None]
    decoder_executed: bool


def _tensor(value, shape, label):
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.float32
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{label} must be finite float32 {shape}")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _output(outputs, name, shape):
    if not isinstance(outputs, Mapping) or name not in outputs:
        raise ValueError(f"Missing required named model output {name!r}")
    return _tensor(outputs[name], shape, name)


class ParaformerPipeline:
    """Compose encoder → predictor → CPU CIF → decoder → greedy text."""

    def __init__(self, encoder, predictor, decoder, names, vocabulary):
        if not all(callable(runner) for runner in (encoder, predictor, decoder)):
            raise TypeError("Each model requires its own callable runner")
        if not isinstance(names, TensorNames):
            raise TypeError("Expected explicit TensorNames from model binding")
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.names = names
        self.vocabulary = validate_vocabulary(vocabulary)

    def predict(self, features, feature_length):
        """Consume prepared features; report only execution/CPU bridge timings.

        Zero CIF tokens bypass decoder and produce empty text. Its timing is
        None, not a measured zero. No frontend, loading or file-I/O latency is
        included. Exceptions propagate without producing a successful result.
        """
        features = _tensor(features, (1, 400, 560), "frontend features")
        if (
            isinstance(feature_length, (bool, np.bool_))
            or not isinstance(feature_length, Integral)
            or not 1 <= feature_length <= 400
        ):
            raise ValueError("feature_length must be an integer in [1,400]")
        names = self.names
        timings = {}
        start = perf_counter()
        raw = self.encoder({names.encoder_input: features})
        timings["encoder"] = (perf_counter() - start) * 1000
        context = _output(raw, names.encoder_output, (1, 400, 512))

        start = perf_counter()
        raw = self.predictor({names.predictor_input: context.copy()})
        timings["predictor"] = (perf_counter() - start) * 1000
        alphas = _output(raw, names.predictor_alphas, (1, 401))
        hidden = _output(raw, names.predictor_hidden, (1, 401, 512))

        start = perf_counter()
        acoustic, count = cif_numpy(alphas, hidden, real_T=feature_length)
        timings["cif"] = (perf_counter() - start) * 1000
        token_count = int(count[0])
        if token_count == 0:
            timings["decoder"] = None
            return Prediction("", (), 0, timings, False)

        start = perf_counter()
        raw = self.decoder(
            {
                names.decoder_context: context,
                names.decoder_count: count,
                names.decoder_bias: np.zeros((1, 1, 512), np.float32),
                names.decoder_acoustic: acoustic,
            }
        )
        timings["decoder"] = (perf_counter() - start) * 1000
        logits = _output(raw, names.decoder_logits, (1, 100, 8404))
        text, ids = decode_logits(logits, token_count, self.vocabulary)
        return Prediction(text, ids, token_count, timings, True)
