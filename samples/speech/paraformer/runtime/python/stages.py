"""Public per-model stages; CIF remains explicit in application composition.

Prepared tensors and returned arrays own their storage. No SDK loading, file I/O,
metrics or next-model execution is performed here. Instances retain no per-call
context. The injected runner may still require serialized access to SDK buffers.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import wraps
import numpy as np

from samples.speech.paraformer.runtime.python.decoding import (
    decode_logits,
    validate_vocabulary,
)


class StageError(ValueError):
    """Stage/operation attribution with the original exception chained as cause."""

    def __init__(self, stage, operation, error):
        self.stage, self.operation = stage, operation
        super().__init__(f"{stage} {operation} failed: {error}")


def attributed(operation):
    @wraps(operation)
    def execute(self, *args, **kwargs):
        try:
            return operation(self, *args, **kwargs)
        except Exception as error:
            raise StageError(self.stage, operation.__name__, error) from error

    return execute


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

    @attributed
    def forward(self, tensors):
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


class EncoderStage(RawStage):
    stage = "encoder"

    def __init__(self, runner, input_name, output_name):
        self.input_name, self.output_name = input_name, output_name
        super().__init__(
            runner,
            {input_name: ((1, 400, 560), "float32")},
            {output_name: ((1, 400, 512), "float32")},
        )

    @attributed
    def pre_process(self, features):
        """Own prepared finite float32 [1,400,560] LFR/CMVN features; no audio I/O."""
        return PreparedInput(
            {self.input_name: tensor(features, (1, 400, 560), "features")}
        )

    @attributed
    def post_process(self, outputs):
        """Return owned float32 context [1,400,512]; no predictor execution."""
        return self._validate(outputs, self.outputs)[self.output_name]


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

    @attributed
    def pre_process(self, context):
        """Own encoder context float32 [1,400,512]; no normalization or casting."""
        return PreparedInput(
            {self.input_name: tensor(context, (1, 400, 512), "context")}
        )

    @attributed
    def post_process(self, outputs):
        """Return owned (alphas [1,401], hidden [1,401,512]); CIF is a separate step."""
        raw = self._validate(outputs, self.outputs)
        return raw[self.alphas_name], raw[self.hidden_name]


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

    @attributed
    def pre_process(self, context, count, acoustic):
        """Own decoder inputs and snapshot count 0..100 as per-call context.

        Inputs: context float32 [1,400,512], count int32 [1], acoustic float32
        [1,100,512]. Bias is source-compatible zeros float32 [1,1,512].
        Zero-count bypass belongs to pipeline.predict, not this stage's forward.
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

    @attributed
    def post_process(self, outputs, context):
        """Greedy decode logits [1,100,8404] using this call's integer count.

        Returns Decoded with special/BPE markers removed, no CTC repeat collapse.
        Raw outputs remain unchanged; no other model is called.
        """
        raw = self._validate(outputs, self.outputs, optional_count=True)
        text, ids = decode_logits(raw[self.logits_name], context, self.vocabulary)
        return Decoded(text, ids, context)
