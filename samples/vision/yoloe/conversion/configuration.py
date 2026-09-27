# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate target-specific PTQ configurations retaining floating output nodes."""

MARCHES = {"x5": "bayes-e", "s100": "nash-e", "s100p": "nash-m"}
ATTENTION_NODE = "/model.10/m/m.0/attn/Softmax"


def make_config(selection, root, graph):
    """Keep source compile policies but omit all output-removal requests.

    Floating output precision is an intent until inspected on the compiled model.
    The variant name is caller-declared; graph shapes do not certify architecture.
    """
    target = selection.target
    march = MARCHES[target]
    prefix = (
        f'yoloe_{selection.variant}_seg_pf_{march.replace("-", "")}_640x640_nv12_float'
    )
    model = {
        "onnx_model": str(root / "source/model.onnx"),
        "march": march,
        "working_dir": str(root / "compiler_output"),
        "output_model_file_prefix": prefix,
    }
    inputs = {
        "input_name": "images",
        "input_shape": "1x3x640x640",
        "input_type_rt": "nv12",
        "input_type_train": "rgb",
        "input_layout_train": "NCHW",
        "scale_value": 1 / 255,
    }
    calibration = {
        "cal_data_dir": str(root / "calibration"),
        "cal_data_type": "float32",
    }
    compiler = {"jobs": 4, "compile_mode": "latency", "core_num": 1}
    warnings = []
    if target == "x5":
        inputs["norm_type"] = "data_scale"
        calibration["calibration_type"] = "default"
        calibration["preprocess_on"] = False
        compiler.update(
            debug=False, optimize_level="O3", input_source={"images": "pyramid"}
        )
        model["layer_out_dump"] = False
        attention_nodes = [ATTENTION_NODE]
        if selection.variant == "11l":
            attention_nodes.append("/model.10/m/m.1/attn/Softmax")
        overrides = {}
        for name in attention_nodes:
            if name in graph["softmax_nodes"]:
                overrides[name] = {
                    "ON": "BPU",
                    "InputType": "int16",
                    "OutputType": "int16",
                }
            else:
                warnings.append(f"source attention node absent: {name}")
        if overrides:
            model["node_info"] = overrides
    elif selection.variant.startswith("26"):
        calibration.update(
            calibration_type="kl",
            quant_config={"model_config": {"all_node_type": "int8"}},
        )
        compiler.update(
            extra_params={"input_no_padding": True, "output_no_padding": False},
            debug=True,
            optimize_level="O2",
        )
    else:
        # The source S11 YAML's explicit removal names belong to v8. Neither
        # those names nor corrected output removal names belong in a float route.
        inputs["norm_type"] = "data_scale"
        calibration.update(
            calibration_type="default",
            quant_config={"op_config": {"softmax": {"qtype": "int8"}}},
        )
        compiler.update(
            extra_params={"input_no_padding": True, "output_no_padding": True},
            debug=True,
            advice=1,
            optimize_level="O2",
            jobs=15,
        )
    return {
        "model_parameters": model,
        "input_parameters": inputs,
        "calibration_parameters": calibration,
        "compiler_parameters": compiler,
    }, warnings
