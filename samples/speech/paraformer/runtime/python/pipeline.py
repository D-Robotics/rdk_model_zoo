# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Three-model Paraformer composition: stages, binding, runners and decode.

``pipeline.py`` owns the model side end to end: the per-model stages
(``EncoderStage``/``PredictorStage``/``DecoderStage`` with the canonical
``preprocess``/``infer``/``postprocess`` spellings), the published tensor
contracts and stage-IO binding, the three lazy raw runners, the greedy
decoder/vocabulary validation and the application-level
:class:`ParaformerPipeline` with its explicit CPU CIF bridge. Audio/file
I/O, published selection and reporting live in ``cli.py``; feature
preparation lives in ``frontend.py`` and the CIF algorithm in ``cif.py``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import wraps
from numbers import Integral
from time import perf_counter

import numpy as np

from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import NamedArrayRunner
from samples.speech.paraformer.runtime.python.cif import cif_numpy
from samples.speech.paraformer.runtime.python.cli import (
    CONTEXT,
    STAGES,
    Selection,
    resolve_selections,
)

# ======================================================================
# Published tensor contracts (names/shapes follow the archived native
# lookup and ONNX extraction graphs).
# ======================================================================

INPUTS = {
    "encoder": {"features": (("speech",), (1, 400, 560), "float32")},
    "predictor": {"context": ((CONTEXT,), (1, 400, 512), "float32")},
    "decoder": {
        "context": ((CONTEXT,), (1, 400, 512), "float32"),
        "count": (("token_num",), (1,), "int32"),
        "bias": (("bias_embed",), (1, 1, 512), "float32"),
        "acoustic": (("onnx::Shape_8609", "shape_8609"), (1, 100, 512), "float32"),
    },
}


OUTPUTS = {
    "encoder": {"context": ((CONTEXT,), (1, 400, 512), "float32")},
    "predictor": {
        "alphas": (("/predictor/Add_output_0",), (1, 401), "float32"),
        "hidden": (("/predictor/Concat_5_output_0",), (1, 401, 512), "float32"),
    },
    "decoder": {"logits": (("logits",), (1, 100, 8404), "float32")},
}


@dataclass(frozen=True)
class Binding:
    selection: Selection
    metadata: RuntimeMetadata
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]

    @property
    def model_name(self):
        return self.metadata.model_name


def _roles(names, shapes, dtypes, contracts):
    if len(set(names)) != len(names):
        raise MetadataMismatchError("Physical tensor names must be unique")
    roles = {}
    for role, (aliases, shape, dtype) in contracts.items():
        matches = [name for name in names if name in aliases]
        if len(matches) != 1:
            raise MetadataMismatchError(f"Expected one exact physical name for {role}")
        name = matches[0]
        if shapes.get(name) != shape or dtypes.get(name) != dtype:
            raise MetadataMismatchError(f"{name} must be {dtype} {shape}")
        roles[role] = name
    if set(names) != set(roles.values()):
        raise MetadataMismatchError("Unexpected extra physical tensors")
    return roles


def validate_selection(selection):
    """Reject stage/asset/path mismatches before SDK construction."""
    if selection.stage not in STAGES:
        raise ValueError("Unknown Paraformer stage")
    expected = resolve_selections(selection.target)[STAGES.index(selection.stage)]
    if selection.asset != expected.asset or (
        not selection.explicit_model_path
        and selection.model_path != expected.model_path
    ):
        raise ValueError("Selection does not match the declared stage publication/path")


def bind_model(selection, metadata):
    """Validate publication identity, one model and every exposed I/O tensor."""
    validate_selection(selection)
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError(
            "Each Paraformer artifact must expose exactly one model"
        )
    inputs, outputs = bind_stage_io(selection.stage, meta)
    return Binding(selection, meta, inputs, outputs)


def bind_stage_io(stage, metadata):
    """Bind physical stage tensors without claiming a published HBM identity."""
    if stage not in STAGES:
        raise ValueError("Unknown Paraformer stage")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    inputs = _roles(
        meta.input_names, meta.input_shapes, meta.input_dtypes, INPUTS[stage]
    )
    output_contract = dict(OUTPUTS[stage])
    # The source ONNX decoder also exposes token_num. Some compiled interfaces
    # eliminate that pass-through output. Validate it when present, never by position.
    if stage == "decoder" and "token_num" in meta.output_names:
        output_contract["count"] = (("token_num",), (1,), "int32")
    outputs = _roles(
        meta.output_names, meta.output_shapes, meta.output_dtypes, output_contract
    )
    return inputs, outputs


def physical_inputs(binding):
    return {
        name: (binding.metadata.input_shapes[name], binding.metadata.input_dtypes[name])
        for name in binding.metadata.input_names
    }

# ======================================================================
# Greedy token decoding and vocabulary validation.
# ======================================================================

def validate_vocabulary(vocabulary):
    """Freeze the ordered vocabulary required by the 8404-class decoder."""
    if (
        not isinstance(vocabulary, (list, tuple))
        or len(vocabulary) != 8404
        or any(not isinstance(token, str) or not token for token in vocabulary)
        or len(set(vocabulary)) != 8404
    ):
        raise ValueError("Expected 8404 unique nonempty ordered vocabulary tokens")
    return tuple(vocabulary)


def decode_logits(logits, token_count, vocabulary):
    """Return text and all selected IDs, including filtered special-token IDs."""
    tokens = validate_vocabulary(vocabulary)
    if (
        not isinstance(logits, np.ndarray)
        or logits.shape != (1, 100, 8404)
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite float32 decoder logits [1,100,8404]")
    if type(token_count) is not int or not 0 <= token_count <= 100:
        raise ValueError("token_count must be an integer in [0,100]")
    ids = tuple(int(i) for i in np.argmax(logits[0, :token_count], axis=-1))
    words = [tokens[i] for i in ids]
    text = "".join(
        token.replace("@@", "")
        for token in words
        if not (token.startswith("<") and token.endswith(">"))
    )
    return text, ids

# ======================================================================
# Public per-model stages; CIF remains explicit in application composition.
# ======================================================================

class StageError(ValueError):
    """Stage/operation attribution with the original exception chained as cause."""

    def __init__(self, stage, operation, error):
        self.stage, self.operation = stage, operation
        super().__init__(f"{stage} {operation} failed: {error}")


def attributed(label):
    """Decorate one stage operation; ``label`` pins the established error wording."""

    def decorate(operation):
        @wraps(operation)
        def execute(self, *args, **kwargs):
            try:
                return operation(self, *args, **kwargs)
            except Exception as error:
                raise StageError(self.stage, label, error) from error

        return execute

    return decorate


def tensor(value, shape, name, dtype="float32"):
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype(dtype)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite {dtype} {shape}")
    return np.array(value, order="C", copy=True)


@dataclass(frozen=True)
class PreparedInput:
    tensors: dict[str, np.ndarray]
    context: int | None = None


@dataclass(frozen=True)
class Decoded:
    text: str
    token_ids: tuple[int, ...]
    token_count: int


class RawStage:
    """One raw model call, owned array adaptation and exact required I/O checks."""

    def __init__(self, runner, inputs, outputs):
        if not callable(runner):
            raise TypeError("runner must be callable")
        expected = {"encoder": (1, 1), "predictor": (1, 2), "decoder": (4, 1)}
        if (len(inputs), len(outputs)) != expected[self.stage] or any(
            not isinstance(name, str) or not name for name in (*inputs, *outputs)
        ):
            raise ValueError(
                "Each stage role requires a distinct nonempty physical name"
            )
        self.runner, self.inputs, self.outputs = runner, inputs, outputs

    def _validate(self, values, contracts, *, optional_count=False):
        if not isinstance(values, Mapping):
            raise ValueError("Expected a named tensor mapping")
        accepted = dict(contracts)
        if optional_count and "token_num" in values and "token_num" not in accepted:
            accepted["token_num"] = ((1,), "int32")
        if set(values) != set(accepted):
            raise ValueError(f"Expected exactly tensor names {tuple(accepted)}")
        return {
            name: tensor(values[name], shape, name, dtype)
            for name, (shape, dtype) in accepted.items()
        }

    @attributed(label="forward")
    def infer(self, tensors):
        """Validate physical inputs, call only this model, return owned raw values.

        No activation, CIF, greedy decoding or file access occurs. Float tensors
        must be finite float32; token count is int32 [1]. Metadata contracts are
        bound by the SDK/ONNX adapter before construction of the pipeline.
        """
        feed = self._validate(tensors, self.inputs)
        if self.stage == "decoder" and not 0 <= int(feed[self.count_name][0]) <= 100:
            raise ValueError("token count must be in [0,100]")
        return self._validate(
            self.runner(feed), self.outputs, optional_count=self.stage == "decoder"
        )

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer` (error wording unchanged)."""
        return self.infer(tensors)


class EncoderStage(RawStage):
    stage = "encoder"

    def __init__(self, runner, input_name, output_name):
        self.input_name, self.output_name = input_name, output_name
        super().__init__(
            runner,
            {input_name: ((1, 400, 560), "float32")},
            {output_name: ((1, 400, 512), "float32")},
        )

    @attributed(label="pre_process")
    def preprocess(self, features):
        """Own prepared finite float32 [1,400,560] LFR/CMVN features; no audio I/O."""
        return PreparedInput(
            {self.input_name: tensor(features, (1, 400, 560), "features")}
        )

    @attributed(label="post_process")
    def postprocess(self, outputs):
        """Return owned float32 context [1,400,512]; no predictor execution."""
        return self._validate(outputs, self.outputs)[self.output_name]

    def pre_process(self, features):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(features)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)


class PredictorStage(RawStage):
    stage = "predictor"

    def __init__(self, runner, input_name, alphas_name, hidden_name):
        self.input_name, self.alphas_name, self.hidden_name = (
            input_name,
            alphas_name,
            hidden_name,
        )
        super().__init__(
            runner,
            {input_name: ((1, 400, 512), "float32")},
            {
                alphas_name: ((1, 401), "float32"),
                hidden_name: ((1, 401, 512), "float32"),
            },
        )

    @attributed(label="pre_process")
    def preprocess(self, context):
        """Own encoder context float32 [1,400,512]; no normalization or casting."""
        return PreparedInput(
            {self.input_name: tensor(context, (1, 400, 512), "context")}
        )

    @attributed(label="post_process")
    def postprocess(self, outputs):
        """Return owned (alphas [1,401], hidden [1,401,512]); CIF is a separate step."""
        raw = self._validate(outputs, self.outputs)
        return raw[self.alphas_name], raw[self.hidden_name]

    def pre_process(self, context):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(context)

    def post_process(self, outputs):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs)


class DecoderStage(RawStage):
    stage = "decoder"

    def __init__(
        self,
        runner,
        context_name,
        count_name,
        bias_name,
        acoustic_name,
        logits_name,
        vocabulary,
    ):
        (
            self.context_name,
            self.count_name,
            self.bias_name,
            self.acoustic_name,
            self.logits_name,
        ) = (context_name, count_name, bias_name, acoustic_name, logits_name)
        self.vocabulary = validate_vocabulary(vocabulary)
        super().__init__(
            runner,
            {
                context_name: ((1, 400, 512), "float32"),
                count_name: ((1,), "int32"),
                bias_name: ((1, 1, 512), "float32"),
                acoustic_name: ((1, 100, 512), "float32"),
            },
            {logits_name: ((1, 100, 8404), "float32")},
        )

    @attributed(label="pre_process")
    def preprocess(self, context, count, acoustic):
        """Own decoder inputs and snapshot count 0..100 as per-call context.

        Inputs: context float32 [1,400,512], count int32 [1], acoustic float32
        [1,100,512]. Bias is source-compatible zeros float32 [1,1,512].
        Zero-count bypass belongs to pipeline.predict, not this stage's infer.
        """
        values = {
            self.context_name: context,
            self.count_name: count,
            self.bias_name: np.zeros((1, 1, 512), np.float32),
            self.acoustic_name: acoustic,
        }
        values = self._validate(values, self.inputs)
        length = int(values[self.count_name][0])
        if not 0 <= length <= 100:
            raise ValueError("token count must be in [0,100]")
        return PreparedInput(values, length)

    @attributed(label="post_process")
    def postprocess(self, outputs, context):
        """Greedy decode logits [1,100,8404] using this call's integer count.

        Returns Decoded with special/BPE markers removed, no CTC repeat collapse.
        Raw outputs remain unchanged; no other model is called.
        """
        raw = self._validate(outputs, self.outputs, optional_count=True)
        text, ids = decode_logits(raw[self.logits_name], context, self.vocabulary)
        return Decoded(text, ids, context)

    def pre_process(self, context, count, acoustic):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(context, count, acoustic)

    def post_process(self, outputs, context):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs, context)

# ======================================================================
# Three shared raw runners with explicit model binding.
# ======================================================================

def load_model_runners(selections, vocabulary, *, runtime_factory=None):
    """Validate and load the three physical model contracts.

    Args:
        selections: Ordered encoder, predictor and decoder selections for S100.
        vocabulary: Ordered 8404-token decoder vocabulary, validated before load.
        runtime_factory: Optional injected SDK factory accepting a model path.

    Returns:
        tuple: Three loaded NamedArrayRunner instances and their TensorNames.

    Raises:
        ValueError: A selection, vocabulary or tensor contract is invalid.
        RuntimeError: Board SDK loading fails. Normal loading checks board
            identity and asset files before constructing the SDK.
    """
    vocabulary = validate_vocabulary(vocabulary)
    if tuple(s.stage for s in selections) != STAGES:
        raise ValueError("Expected ordered encoder, predictor and decoder selections")
    if any(s.target != "s100" for s in selections):
        raise ValueError("All three models must target s100")
    for selection in selections:
        validate_selection(selection)
    runners = tuple(
        NamedArrayRunner(
            selection,
            binding_loader=bind_model,
            physical_inputs=physical_inputs,
            task_name=f"Paraformer {selection.stage}",
            runtime_factory=runtime_factory,
        )
        for selection in selections
    )
    encoder, predictor, decoder = (runner.load() for runner in runners)
    from samples.speech.paraformer.runtime.python.pipeline import TensorNames

    names = TensorNames(
        encoder_input=encoder.inputs["features"],
        encoder_output=encoder.outputs["context"],
        predictor_input=predictor.inputs["context"],
        predictor_alphas=predictor.outputs["alphas"],
        predictor_hidden=predictor.outputs["hidden"],
        decoder_context=decoder.inputs["context"],
        decoder_count=decoder.inputs["count"],
        decoder_bias=decoder.inputs["bias"],
        decoder_acoustic=decoder.inputs["acoustic"],
        decoder_logits=decoder.outputs["logits"],
    )
    return runners, names

# ======================================================================
# Application-level three-model composition, with an explicit CPU CIF
# bridge.
# ======================================================================

"""Application-level three-model composition, with an explicit CPU CIF bridge.

Each injected callable performs one raw model execution with named tensors.
The pipeline initializes the three model runtimes through its loader and owns
scheduling. Audio/file I/O and reporting belong to the CLI and frontend modules.
"""

from dataclasses import dataclass, fields
from numbers import Integral
from time import perf_counter

import numpy as np



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
    """Compose encoder → predictor → CPU CIF → decoder → greedy text.

    Attributes:
        runners: Ordered encoder, predictor and decoder callables. Pipelines
            created by from_models own three loaded NamedArrayRunner instances.
        names: Bound physical tensor names for the complete model group.
        vocabulary: Frozen ordered decoder token strings.
        encoder_stage: Feature-to-context stage using float32 [1,400,560] input.
        predictor_stage: Context-to-weight/hidden stage for the CPU CIF bridge.
        decoder_stage: Context/acoustic/count-to-token stage.
    """

    def __init__(self, encoder, predictor, decoder, names, vocabulary):
        if not all(callable(runner) for runner in (encoder, predictor, decoder)):
            raise TypeError("Each model requires its own callable runner")
        if not isinstance(names, TensorNames):
            raise TypeError("Expected explicit TensorNames from model binding")
        self.runners = (encoder, predictor, decoder)
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

    @classmethod
    def from_models(cls, selections, vocabulary, *, runtime_factory=None):
        """Load and bind the encoder, predictor and decoder for inference.

        Args:
            selections: Ordered encoder, predictor and decoder ModelSelections
                targeting S100. Normal loading checks board and model identity.
            vocabulary: Ordered sequence of 8404 decoder token strings.
            runtime_factory: Optional SDK factory used for injected runtimes.

        Returns:
            ParaformerPipeline: A loaded pipeline ready for predict.

        Raises:
            ValueError: Model selections, vocabulary or physical tensor metadata
                do not match the declared three-model contract.
            RuntimeError: SDK construction or model loading fails.
        """

        runners, names = load_model_runners(
            selections, vocabulary, runtime_factory=runtime_factory
        )
        return cls(*runners, names, vocabulary)

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Apply model-keyed scheduling to all three loaded runtimes.

        Args:
            priority: Optional integer SDK scheduling priority in [0,255].
            bpu_cores: Optional list of nonnegative BPU core indexes.

        Raises:
            RuntimeError: Any runtime lacks a scheduling setter or rejects it.
            ValueError: Scheduling values are invalid. SDK failures propagate;
                earlier applied settings are retained if a later setter fails.
        """
        if priority is None and bpu_cores is None:
            return
        for runner in self.runners:
            if not callable(getattr(runner.runtime, "set_scheduling_params", None)):
                raise RuntimeError(
                    "Every model runtime must expose set_scheduling_params"
                )
        for runner in self.runners:
            runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

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
        prepared = self.encoder_stage.preprocess(features)
        start = perf_counter()
        raw = self.encoder_stage.infer(prepared.tensors)
        timings["encoder"] = (perf_counter() - start) * 1000
        context = self.encoder_stage.postprocess(raw)

        prepared = self.predictor_stage.preprocess(context)
        start = perf_counter()
        raw = self.predictor_stage.infer(prepared.tensors)
        timings["predictor"] = (perf_counter() - start) * 1000
        alphas, hidden = self.predictor_stage.postprocess(raw)

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

        prepared = self.decoder_stage.preprocess(context, count, acoustic)
        start = perf_counter()
        raw = self.decoder_stage.infer(prepared.tensors)
        timings["decoder"] = (perf_counter() - start) * 1000
        decoded = self.decoder_stage.postprocess(raw, prepared.context)
        return Prediction(
            decoded.text, decoded.token_ids, decoded.token_count, timings, True
        )
