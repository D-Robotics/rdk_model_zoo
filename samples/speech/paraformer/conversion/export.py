"""Export the three fixed-shape Paraformer stages on CPU, with numeric checks."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.py_utils.assets import sha256_file
from samples.speech.paraformer.runtime.python.cli import (
    LOCAL_DIGESTS,
    VOCABULARY_DIGEST,
    write_json,
)
from samples.speech.paraformer.runtime.python.pipeline import INPUTS, OUTPUTS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local model.pt/config.yaml/tokens.json/am.mvn directory; never downloaded",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New export directory"
    )
    parser.add_argument(
        "--feature",
        type=Path,
        action="append",
        default=[],
        help="Additional real prepared float32 [1,400,560] NPY, repeatable",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="CPU threads (default: 4)"
    )
    return parser


def preflight(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("Output directory must be new")
    source = args.model_dir.expanduser().resolve()
    files = {
        name: source / name
        for name in ("model.pt", "config.yaml", "tokens.json", "am.mvn")
    }
    for path in (*files.values(), *args.feature):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty input: {path}")
    identities = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in files.items()
    }
    for name, digest in {
        "config.yaml": LOCAL_DIGESTS["paraformer_config.yaml"],
        "am.mvn": LOCAL_DIGESTS["am.mvn"],
        "tokens.json": VOCABULARY_DIGEST,
    }.items():
        if identities[name]["sha256"] != digest:
            raise ValueError(f"Expected the pinned Paraformer {name}")
    return files, identities


def load_model(files):
    import torch
    import yaml
    from funasr.models.contextual_paraformer.model import ContextualParaformer

    config = yaml.safe_load(files["config.yaml"].read_text())
    kwargs = dict(config)
    kwargs.update(config["model_conf"])
    kwargs.update(input_size=560, vocab_size=8404)
    model = ContextualParaformer(**kwargs).eval()
    # Strict loading prevents upstream's missing-key fallback to random parameters.
    state = torch.load(files["model.pt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    for name, tensor in model.state_dict().items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite model parameter: {name}")
    return model


def signature(stage, side):
    return [
        (aliases[0], shape, dtype)
        for aliases, shape, dtype in (INPUTS if side == "input" else OUTPUTS)[
            stage
        ].values()
    ]


def bind_output_names(model, stage):
    """Bind wrapper tuple positions, preserving colliding internal tensor uses."""
    expected = [name for name, _, _ in signature(stage, "output")]
    if len(model.graph.output) != len(expected):
        raise ValueError("Unexpected exported output count")

    def rename(old, new):
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
            *model.graph.initializer,
        ):
            if value.name == old:
                value.name = new
        for node in model.graph.node:
            for side in (node.input, node.output):
                for i, name in enumerate(side):
                    if name == old:
                        side[i] = new

    for index, name in enumerate(expected):
        current = model.graph.output[index].name
        if current == name:
            continue
        used = {
            v.name
            for v in (
                *model.graph.input,
                *model.graph.initializer,
                *model.graph.value_info,
                *model.graph.output,
            )
        }
        used.update(n for node in model.graph.node for n in (*node.input, *node.output))
        if name in used:
            if name in {v.name for v in model.graph.input}:
                raise ValueError("Output contract collides with input contract")
            replacement = "__paraformer_internal_" + str(index)
            while replacement in used:
                replacement += "_"
            rename(name, replacement)
        rename(current, name)
    return model


def check_signature(model, stage):
    import onnx
    import numpy as np

    for side in ("input", "output"):
        values = list(getattr(model.graph, side))
        expected = signature(stage, side)
        if len(values) != len(expected):
            raise ValueError(f"Unexpected {stage} {side} count")
        for value, (name, shape, dtype) in zip(values, expected, strict=True):
            tensor = value.type.tensor_type
            actual = tuple(
                d.dim_value if d.HasField("dim_value") else None
                for d in tensor.shape.dim
            )
            if (
                value.name != name
                or actual != shape
                or onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type)
                != np.dtype(dtype)
            ):
                raise ValueError(f"Unexpected {stage} {side}: {value}")


def compare(reference, actual):
    import numpy as np

    if len(reference) != len(actual):
        raise ValueError("Output count changed")
    differences = []
    for left, right in zip(reference, actual, strict=True):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not np.isfinite(right).all()
        ):
            raise ValueError("Output shape/dtype/finiteness changed")
        np.testing.assert_allclose(right, left, rtol=1e-4, atol=1e-4)
        differences.append(float(np.max(np.abs(left - right))))
    return differences


def export_stage(stage, module, cases, output):
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    from samples.speech.paraformer.conversion.graph_ops import (
        fold_constant_ranges,
        gather_indices_int32,
        normalize_axes,
    )

    input_names = [name for name, _, _ in signature(stage, "input")]
    output_names = [name for name, _, _ in signature(stage, "output")]
    raw_path = output / f"{stage}.raw.onnx"
    final_path = output / f"{stage}.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            module,
            cases[0][1],
            str(raw_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=15,
            do_constant_folding=True,
            dynamo=False,
        )
    graph = bind_output_names(onnx.load(raw_path), stage)
    onnx.checker.check_model(graph)
    check_signature(graph, stage)
    nodes_before = len(graph.graph.node)
    # No data-dependent Range folding, no implicit dynamic index narrowing.
    for operation in (fold_constant_ranges, gather_indices_int32, normalize_axes):
        graph = operation(graph)
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.checker.check_model(graph)
    check_signature(graph, stage)
    onnx.save(graph, final_path)
    nodes_after = len(graph.graph.node)
    del graph
    gc.collect()
    options = ort.SessionOptions()
    options.intra_op_num_threads = torch.get_num_threads()
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(final_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    checks = []
    with torch.inference_mode():
        for label, args in cases:
            reference = module(*args)
            reference = reference if isinstance(reference, tuple) else (reference,)
            expected = [x.detach().cpu().numpy() for x in reference]
            feeds = {
                name: value.detach().cpu().numpy()
                for name, value in zip(input_names, args, strict=True)
            }
            actual = session.run(output_names, feeds)
            checks.append({"case": label, "max_abs_diff": compare(expected, actual)})
    return {
        "path": final_path.name,
        "sha256": sha256_file(final_path),
        "raw_sha256": sha256_file(raw_path),
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "rtol": 1e-4,
        "atol": 1e-4,
        "checks": checks,
    }


def execute(args):
    files, identities = preflight(args)
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import funasr
    from samples.speech.paraformer.conversion.torch_stages import build_stages
    from samples.speech.paraformer.runtime.python.cif import cif_numpy

    feature_inputs = []
    for path in args.feature:
        # Read and hash the same bytes; np.load never permits object arrays.
        import hashlib
        import io

        data = path.read_bytes()
        value = np.load(io.BytesIO(data), allow_pickle=False)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (1, 400, 560)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid prepared feature tensor: {path}")
        feature_inputs.append((path, value, hashlib.sha256(data).hexdigest()))
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "rdk-model-zoo/paraformer-export/v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": identities,
        "features": [
            {"path": str(p.resolve()), "sha256": d} for p, _, d in feature_inputs
        ],
        "stages": {},
        "environment": {
            "torch": torch.__version__,
            "funasr": funasr.__version__,
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "scope": "CPU FP32 stage export; OE/SDK/board not-run",
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.threads)
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(191009)
            model = load_model(files)
            stages = build_stages(model)
            del model
            features = [
                ("zeros", torch.zeros(1, 400, 560)),
                ("random", torch.randn(1, 400, 560)),
            ]
            features += [
                (f"feature-{i}", torch.from_numpy(value.copy()))
                for i, (_, value, _) in enumerate(feature_inputs)
            ]
            contexts = [(name, stages["encoder"](value)) for name, value in features]
            acoustic_cases = []
            for name, context in contexts:
                alphas, hidden = stages["predictor"](context)
                # Deliberately unmasked: conversion-stage checks, not an utterance CER run.
                acoustic, count = cif_numpy(alphas.numpy(), hidden.numpy(), real_T=None)
                acoustic_cases.append(
                    (
                        name,
                        (
                            context,
                            torch.from_numpy(count),
                            torch.zeros(1, 1, 512),
                            torch.from_numpy(acoustic),
                        ),
                    )
                )
            for count in (0, 1, 17, 100):
                acoustic_cases.append(
                    (
                        f"count-{count}",
                        (
                            contexts[0][1],
                            torch.tensor([count], dtype=torch.int32),
                            torch.zeros(1, 1, 512),
                            torch.randn(1, 100, 512),
                        ),
                    )
                )
            cases = {
                "encoder": [(n, (v,)) for n, v in features],
                "predictor": [(n, (v,)) for n, v in contexts],
                "decoder": acoustic_cases,
            }
            for stage in ("encoder", "predictor", "decoder"):
                print(f"Exporting and checking {stage}", flush=True)
                report["stages"][stage] = export_stage(
                    stage, stages[stage], cases[stage], output
                )
                write_json(output / "export-report.json", report)
            for name, path in files.items():
                if sha256_file(path) != identities[name]["sha256"]:
                    raise ValueError(f"Source changed during export: {path}")
            report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        torch.set_num_threads(previous_threads)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "export-report.json", report)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = execute(args)
    except Exception as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
