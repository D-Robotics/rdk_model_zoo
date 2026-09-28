"""S100/nash-e source recipe settings with one consistent workspace layout."""

from samples.speech.paraformer.conversion.export import signature

STAGES = ("encoder", "predictor", "decoder")
CALIBRATION_INPUTS = {
    "encoder": ("speech",),
    "predictor": ("encoder_after_norm_Add_1_output_0",),
    "decoder": (
        "encoder_after_norm_Add_1_output_0",
        "token_num",
        "bias_embed",
        "shape_8609",
    ),
}
PREFIXES = {
    "encoder": "paraformer_encoder_int16",
    "predictor": "predictor_int16",
    "decoder": "decoder_int16",
}


def make_config(stage, jobs=32):
    if stage not in STAGES or type(jobs) is not int or jobs < 1:
        raise ValueError("Expected a known stage and positive compiler jobs")
    tensors = signature(stage, "input")
    return {
        "model_parameters": {
            "onnx_model": f"source/{stage}.onnx",
            "march": "nash-e",
            "output_model_file_prefix": PREFIXES[stage],
            "working_dir": f"compiled/{stage}",
        },
        "input_parameters": {
            "input_name": ";".join(name for name, _, _ in tensors),
            "input_shape": ";".join(
                "x".join(map(str, shape)) for _, shape, _ in tensors
            ),
            "input_type_rt": ";".join(["featuremap"] * len(tensors)),
            "input_type_train": ";".join(["featuremap"] * len(tensors)),
            "input_layout_train": ";".join(["NCHW"] * len(tensors)),
            "separate_batch": False,
        },
        "calibration_parameters": {
            "cal_data_dir": ";".join(
                f"calibration/{name}" for name in CALIBRATION_INPUTS[stage]
            ),
            "calibration_type": "max",
            "quant_config": {
                "model_config": {
                    "all_node_type": "int16",
                    "activation": {"calibration_type": "max"},
                }
            },
        },
        "compiler_parameters": {
            "optimize_level": "O2",
            "compile_mode": "latency",
            "core_num": 1,
            "jobs": jobs,
            "cache_mode": "disable",
        },
    }
