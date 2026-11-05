"""Prepare the recorded x1.0 RGB conversion, or derive its late-INT8 config.

Run in an existing X5 OpenExplorer environment. Compilation remains an explicit
native hb_mapper command; this script does not install dependencies.
"""

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from samples.vision.osnet.evaluator.metrics import records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    if digest(args.source_file) != "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae":
        raise ValueError("Source differs from the recorded OSNet experiment")
    import numpy as np
    import onnx
    import onnxruntime as ort
    from PIL import Image
    import torch
    from torch import nn
    import yaml

    if digest(args.weights) != "2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154":
        raise ValueError("Expected the recorded official x1.0 Market1501 weights")
    workspace = args.workspace.resolve()
    export_only = args.command in ("export", "export-s")
    opset = 19 if args.command == "export-s" else 11
    train = [] if export_only else records(args.dataset / "bounding_box_train")
    query = records(args.dataset / "query")
    gallery = records(args.dataset / "bounding_box_test")
    if (len(query), len(gallery)) != (3368, 15913) or (not export_only and len(train) != 12936):
        raise ValueError("Expected the complete Market1501 splits")
    if {r[1] for r in train} & {r[1] for r in query}:
        raise ValueError("Calibration identities overlap the query split")
    calibration = workspace / "data/calibration_rgb_nchw"
    if not export_only:
        calibration.mkdir(parents=True, exist_ok=False)
    weights_dir = workspace / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = weights_dir / "osnet_x1_0_spatial.onnx"
    if onnx_path.exists():
        raise FileExistsError(onnx_path)
    selected = [] if export_only else np.random.default_rng(20261029).choice(len(train), size=128, replace=False)
    entries = []
    for index in selected:
        path, pid, camera = train[index]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").resize((128, 256), Image.Resampling.BILINEAR), np.float32)
        output = calibration / (path.stem + ".bin")
        np.ascontiguousarray(rgb.transpose(2, 0, 1)).tofile(output)
        entries.append({"image": str(path), "image_sha256": digest(path), "tensor_sha256": digest(output), "pid": pid, "camera": camera})
    if not export_only:
        (workspace / "data/calibration_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("official_osnet", args.source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.osnet_x1_0(num_classes=751, pretrained=False).eval()
    original.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True), strict=True)

    class SpatialHead(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.backbone = source
            self.projection = nn.Conv2d(512, 512, 1, bias=True)
            self.projection.weight.data.copy_(source.fc[0].weight[:, :, None, None])
            self.projection.bias.data.copy_(source.fc[0].bias)
            self.bn = nn.BatchNorm2d(512, eps=source.fc[1].eps)
            self.bn.load_state_dict(source.fc[1].state_dict())
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.backbone.global_avgpool(self.backbone.featuremaps(x))
            return self.relu(self.bn(self.projection(x)))

    def preprocess(path):
        with Image.open(path) as image:
            x = np.asarray(image.convert("RGB").resize((128, 256), Image.Resampling.BILINEAR), np.float32).transpose(2, 0, 1) / np.float32(255)
        return np.ascontiguousarray(((x - np.array([.485, .456, .406], np.float32)[:, None, None]) /
                                   np.array([.229, .224, .225], np.float32)[:, None, None])[None])

    adapted = SpatialHead(original).eval()
    torch.set_num_threads(4)
    exporter_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.inference_mode():
        torch.onnx.export(adapted, torch.from_numpy(preprocess(query[0][0])), str(onnx_path),
                          opset_version=opset, input_names=["images"], output_names=["embedding"], do_constant_folding=True, **exporter_options)
    onnx.checker.check_model(onnx.load(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    errors = []
    with torch.inference_mode():
        for path, _, _ in query[::200][:16]:
            x = preprocess(path)
            expected = original(torch.from_numpy(x)).numpy()
            actual = session.run(None, {"images": x})[0].reshape(1, 512)
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            errors.append(float(np.max(np.abs(actual - expected))))
    if not export_only:
        base = {
            "model_parameters": {"onnx_model": str(onnx_path), "march": "bayes-e", "output_model_file_prefix": "osnet_x1_0", "layer_out_dump": False},
            "input_parameters": {"input_name": "images", "input_shape": "1x3x256x128", "input_type_train": "rgb", "input_layout_train": "NCHW",
                                 "input_type_rt": "rgb", "input_layout_rt": "NHWC", "norm_type": "data_mean_and_scale",
                                 "mean_value": "123.675 116.28 103.53", "scale_value": "0.0171247538316637 0.0175070028011204 0.0174291938997821"},
            "calibration_parameters": {"cal_data_dir": str(calibration), "cal_data_type": "float32", "calibration_type": "default"},
            "compiler_parameters": {"compile_mode": "latency", "optimize_level": "O3", "debug": False, "jobs": 8},
        }
        for variant in ("spatial-rgb", "spatial-rgb-int16"):
            config = copy.deepcopy(base)
            folder = workspace / "runs" / variant
            folder.mkdir(parents=True, exist_ok=False)
            config["model_parameters"]["working_dir"] = str(folder / "compiled")
            if variant.endswith("int16"):
                config["calibration_parameters"].update(optimization="set_all_nodes_int16", per_channel=False)
            (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    receipt = {"source_sha256": digest(args.source_file), "weights_sha256": digest(args.weights), "onnx_sha256": digest(onnx_path),
               "spatial_head_max_abs_error": max(errors), "verification_samples": len(errors), "opset": opset, "board_test": "not-run"}
    (workspace / "runs").mkdir(parents=True, exist_ok=True)
    receipt["purpose"] = "float_baseline_only" if export_only else "conversion"
    receipt["versions"] = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__}
    receipt["providers"] = session.get_providers()
    (workspace / "runs/export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def late8(args):
    import onnx
    import yaml

    workspace = args.workspace.resolve()
    base = workspace / "runs/spatial-rgb-int16"
    config = yaml.safe_load((base / "config.yaml").read_text())
    graph = onnx.load(base / "compiled/osnet_x1_0_optimized_float_model.onnx").graph
    nodes = {node.name: {"InputType": "int8", "OutputType": "int8"} for node in graph.node
             if node.name.startswith(("/conv4/", "/conv5/", "/projection/", "/backbone/conv4/", "/backbone/conv5/")) and node.op_type not in ("Constant", "Identity")}
    if not nodes:
        raise ValueError("No recorded late-stage nodes found; inspect the exported graph")
    folder = workspace / "runs/spatial-rgb-late8"
    folder.mkdir(exist_ok=False)
    config["model_parameters"].update(working_dir=str(folder / "compiled"), node_info=nodes)
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(folder / "config.yaml")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "export", "export-s"):
        prepare_parser = commands.add_parser(command, help="Export and verify FP32; prepare also creates training-only calibration and PTQ configs")
        prepare_parser.add_argument("--source-file", type=Path, default=Path(__file__).resolve().parents[1] / "third_party/osnet.py", help="Pinned upstream OSNet source")
        prepare_parser.add_argument("--weights", type=Path, required=True)
        prepare_parser.add_argument("--dataset", type=Path, required=True)
        prepare_parser.add_argument("--workspace", type=Path, required=True)
    mixed_parser = commands.add_parser("late8", help="Derive the recorded mixed-precision configuration after INT16 compilation")
    mixed_parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    {"prepare": prepare, "export": prepare, "export-s": prepare, "late8": late8}[args.command](args)


if __name__ == "__main__":
    main()
