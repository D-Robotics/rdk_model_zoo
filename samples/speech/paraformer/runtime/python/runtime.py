"""Three shared raw runners, explicit model binding and real SDK scheduling."""

from dataclasses import dataclass

from samples._shared.single_array_runner import NamedArrayRunner
from samples.speech.paraformer.runtime.python.decoding import validate_vocabulary
from samples.speech.paraformer.runtime.python.model_binding import (
    STAGES,
    bind_model,
    physical_inputs,
    validate_selection,
)
from samples.speech.paraformer.runtime.python.pipeline import (
    ParaformerPipeline,
    TensorNames,
)


@dataclass
class RuntimeBundle:
    runners: tuple
    pipeline: ParaformerPipeline

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Delegate to all models, rejecting unsupported SDK setters first."""
        if priority is None and bpu_cores is None:
            return
        for runner in self.runners:
            if not callable(getattr(runner.runtime, "set_scheduling_params", None)):
                raise RuntimeError(
                    "Every model runtime must expose set_scheduling_params"
                )
        # Shared runner validates values and uses model-keyed SDK dictionaries.
        # If the SDK itself raises midway, propagate it; no rollback is promised.
        for runner in self.runners:
            runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)


def load_runtime(selections, vocabulary, *, runtime_factory=None):
    """Load exactly the three selections; injected factory is a host-test seam.

    The normal path uses shared board-identity and asset-file checks before SDK
    construction. It does not download models or infer support from a filename.
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
    return RuntimeBundle(runners, ParaformerPipeline(*runners, names, vocabulary))
