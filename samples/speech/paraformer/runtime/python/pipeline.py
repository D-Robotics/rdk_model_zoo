"""Application-level three-model composition, with an explicit CPU CIF bridge.

Each injected callable performs one raw model execution with named tensors.
SDK loading, physical metadata binding, scheduling, audio/file I/O and reporting
belong to their respective adapters and application entry, not this module.
"""

from dataclasses import dataclass, fields
from numbers import Integral
from time import perf_counter

import numpy as np

from samples.speech.paraformer.runtime.python.cif import cif_numpy
from samples.speech.paraformer.runtime.python.stages import (
    EncoderStage,
    PredictorStage,
    DecoderStage,
    StageError,
)
from samples.speech.paraformer.runtime.python.decoding import (
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
        self.encoder_stage = EncoderStage(
            lambda feed: self.encoder(feed), names.encoder_input, names.encoder_output
        )
        self.predictor_stage = PredictorStage(
            lambda feed: self.predictor(feed),
            names.predictor_input,
            names.predictor_alphas,
            names.predictor_hidden,
        )
        self.decoder_stage = DecoderStage(
            lambda feed: self.decoder(feed),
            names.decoder_context,
            names.decoder_count,
            names.decoder_bias,
            names.decoder_acoustic,
            names.decoder_logits,
            self.vocabulary,
        )

    def predict(self, features, feature_length):
        """Consume prepared features; report only execution/CPU bridge timings.

        Zero CIF tokens bypass decoder and produce empty text. Its timing is
        None, not a measured zero. No frontend, loading or file-I/O latency is
        included. StageError identifies the failing stage/operation and chains its
        original exception; no successful result is produced on failure.
        """
        if (
            isinstance(feature_length, (bool, np.bool_))
            or not isinstance(feature_length, Integral)
            or not 1 <= feature_length <= 400
        ):
            raise ValueError("feature_length must be an integer in [1,400]")
        timings = {}
        prepared = self.encoder_stage.pre_process(features)
        start = perf_counter()
        raw = self.encoder_stage.forward(prepared.tensors)
        timings["encoder"] = (perf_counter() - start) * 1000
        context = self.encoder_stage.post_process(raw)

        prepared = self.predictor_stage.pre_process(context)
        start = perf_counter()
        raw = self.predictor_stage.forward(prepared.tensors)
        timings["predictor"] = (perf_counter() - start) * 1000
        alphas, hidden = self.predictor_stage.post_process(raw)

        start = perf_counter()
        try:
            acoustic, count = cif_numpy(alphas, hidden, real_T=feature_length)
        except Exception as error:
            raise StageError("cif", "integrate", error) from error
        timings["cif"] = (perf_counter() - start) * 1000
        token_count = int(count[0])
        if token_count == 0:
            timings["decoder"] = None
            return Prediction("", (), 0, timings, False)

        prepared = self.decoder_stage.pre_process(context, count, acoustic)
        start = perf_counter()
        raw = self.decoder_stage.forward(prepared.tensors)
        timings["decoder"] = (perf_counter() - start) * 1000
        decoded = self.decoder_stage.post_process(raw, prepared.context)
        return Prediction(
            decoded.text, decoded.token_ids, decoded.token_count, timings, True
        )
