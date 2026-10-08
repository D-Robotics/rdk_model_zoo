"""Three shared raw runners, explicit model binding and real SDK scheduling."""

from utils.py_utils.single_array_runner import NamedArrayRunner
from samples.speech.paraformer.runtime.python.decoding import validate_vocabulary
from samples.speech.paraformer.runtime.python.model_binding import (
    STAGES,
    bind_model,
    physical_inputs,
    validate_selection,
)
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
