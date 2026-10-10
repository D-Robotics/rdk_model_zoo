"""Reproducible MobileNet checkpoint, ONNX, calibration, and evaluation CLI.

Run from any directory. Large inputs and immutable run outputs belong outside
the source checkout. This tool does not compile, publish, or claim board support.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform as host_platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from utils.py_utils.classification_host import (
    calibration_input, float_input, load_dataset, prepare_rgb, sha256_file,
    top5_ids, validate_contract, write_json,
)

PINS = Path(__file__).with_name("checkpoints.json")
MARCH = {"x5": "bayes-e", "s100": "nash-e", "s100p": "nash-m", "s600": "nash-p"}


def environment() -> dict:
    """Return actual dependency versions, interpreter, and OS identity."""
    versions = {}
    for name in ("torch", "torchvision", "timm", "onnx", "onnxruntime-gpu",
                 "onnxruntime", "onnxsim", "numpy", "Pillow", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "executable": sys.executable,
            "platform": host_platform.platform(), "packages": versions}


def provenance() -> dict:
    """Record the Git base and exact host workflow source file hashes."""
    files = [Path(__file__), PINS, ROOT / "utils/py_utils/classification_host.py"]
    return {"base_commit": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        "git_status": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True)}


def receipt(stage: str, key: str, spec: dict) -> dict:
    """Create identity fields shared by workflow receipts.

    Args:
        stage: Actual stage being executed.
        key: Checkpoint matrix key.
        spec: Frozen checkpoint and preprocessing specification.

    Returns:
        Receipt header, without an implied release or board status.
    """
    return {"schema_version": 1, "stage": stage, "model": key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec": spec, "pins_sha256": sha256_file(PINS),
            "source": provenance(), "environment": environment(), "command": sys.argv}


def check_source(source_dir: Path, spec: dict) -> None:
    """Verify pinned local source files before deserialization or inference.

    Args:
        source_dir: Directory containing the downloaded source artifacts.
        spec: Checkpoint entry with immutable file hashes.

    Raises:
        ValueError: Any source file differs from the frozen bytes.
    """
    for filename, metadata in spec["artifacts"].items():
        if sha256_file(source_dir / filename) != metadata["sha256"]:
            raise ValueError(f"Source hash mismatch: {filename}")


def fetch_source(args: argparse.Namespace, spec: dict) -> None:
    """Download pinned public artifacts and verify their hashes.

    Args:
        args: CLI arguments with a new output directory.
        spec: Immutable checkpoint specification.
    """
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, metadata in spec["artifacts"].items():
        path = args.output / filename
        with urllib.request.urlopen(metadata["url"], timeout=120) as response, path.open("xb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"Downloaded hash mismatch: {filename}")
    result = receipt("fetch", args.model, spec)
    result["status"] = "verified"
    write_json(args.output / "fetch.json", result)


def load_torch_model(source_dir: Path, spec: dict):
    """Instantiate a local model with strict safetensors loading, offline.

    Args:
        source_dir: Verified checkpoint directory.
        spec: Architecture and weight identity.

    Returns:
        CPU model in evaluation mode, with no network weight resolution.
    """
    import timm
    from safetensors.torch import load_file

    check_source(source_dir, spec)
    model = timm.create_model(spec["architecture"], pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(str(source_dir / "model.safetensors")), strict=True)
    return model.eval()


def make_session(path: Path, contract: dict, provider: str, threads: int):
    """Create ORT with an explicit provider and validate the tensor contract.

    Args:
        path: FP32 ONNX graph.
        contract: Frozen input/output geometry.
        provider: Explicit ORT execution provider; unavailable GPU is an error.
        threads: Positive CPU thread limit.

    Returns:
        Loaded inference session.

    Raises:
        ValueError: Provider or tensor contract is invalid.
    """
    import onnxruntime as ort

    if provider not in ort.get_available_providers() or threads < 1:
        raise ValueError(f"Unavailable provider or invalid thread count: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    if session.get_providers()[0] != provider:
        raise ValueError("Requested provider failed to initialize")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    shape = [1, 3, contract["size"], contract["size"]]
    if len(inputs) != 1 or inputs[0].shape != shape or inputs[0].type != "tensor(float)":
        raise ValueError("ONNX input contract mismatch")
    if len(outputs) != 1 or outputs[0].shape != [1, 1000] or outputs[0].type != "tensor(float)":
        raise ValueError("ONNX output contract mismatch")
    return session


def verify_export(export_dir: Path, key: str, spec: dict) -> tuple[Path, dict]:
    """Bind a graph to its passing export receipt and current contract.

    Args:
        export_dir: Completed export directory.
        key: Expected model key.
        spec: Current pinned model contract.

    Returns:
        Verified graph path and export receipt.

    Raises:
        ValueError: Graph identity, receipt, or contract differs.
    """
    metadata = json.loads((export_dir / "export.json").read_text())
    graph = export_dir / "model.onnx"
    if (metadata["status"] != "passed" or metadata["model"] != key
            or metadata["spec"] != spec or sha256_file(graph) != metadata["onnx_sha256"]):
        raise ValueError("Export receipt/model/contract mismatch")
    return graph, metadata


def export_model(args: argparse.Namespace, spec: dict) -> None:
    """Export batch-one ONNX and compare real-image logits with the source.

    Args:
        args: Paths, opset, smoke images, and output directory.
        spec: Frozen source and preprocessing contract.
    """
    import onnx
    import onnxsim
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = load_torch_model(args.source_dir, spec)
    contract = spec["contract"]
    args.output.mkdir(parents=True, exist_ok=False)
    graph_path = args.output / "model.onnx"
    example = torch.zeros(1, 3, contract["size"], contract["size"])
    with torch.inference_mode():
        torch.onnx.export(model, example, str(graph_path), opset_version=args.opset,
                          input_names=["data"], output_names=["logits"],
                          do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(graph_path))
    onnx.checker.check_model(graph)
    if args.simplify:
        graph, valid = onnxsim.simplify(graph)
        if not valid:
            raise ValueError("ONNX simplification check failed")
        onnx.save(graph, str(graph_path))
    onnx.checker.check_model(graph)
    session = make_session(graph_path, contract, "CPUExecutionProvider", args.threads)
    comparisons = []
    for image in args.images:
        data = float_input(prepare_rgb(image, contract), contract)
        with torch.inference_mode():
            expected = model(torch.from_numpy(data)).numpy()
        actual = session.run(None, {session.get_inputs()[0].name: data})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        if top5_ids(actual) != top5_ids(expected):
            raise ValueError(f"Top-5 ordering differs for {image}")
        comparisons.append({"path": str(image.resolve()), "sha256": sha256_file(image),
                            "max_abs_error": float(np.max(np.abs(actual - expected))),
                            "top5": top5_ids(actual)})
    # Profile only Conv/Linear MACs explicitly; do not call this total GFLOPs.
    macs = [0]

    def count_macs(module, inputs, output):
        """Accumulate Conv/Linear multiply-accumulate counts for one forward."""
        if isinstance(module, torch.nn.Conv2d):
            macs[0] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
        elif isinstance(module, torch.nn.Linear):
            macs[0] += output.numel() * module.in_features

    handles = [module.register_forward_hook(count_macs) for module in model.modules()
               if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.inference_mode():
        model(example)
    for handle in handles:
        handle.remove()
    result = receipt("export", args.model, spec)
    result.update(status="passed", onnx_sha256=sha256_file(graph_path),
                  opset=args.opset, simplified=args.simplify, comparisons=comparisons,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  conv_linear_macs=int(macs[0]),
                  compute_scope="Conv/Linear MACs only; not total GFLOPs",
                  input_shape=[1, 3, contract["size"], contract["size"]],
                  output_shape=[1, 1000], external_batch=1)
    write_json(args.output / "export.json", result)


def prepare_calibration(args: argparse.Namespace, spec: dict) -> None:
    """Verify disjoint inputs and materialize platform-specific calibration.

    Args:
        args: Calibration/evaluation manifests, image root, platform, output.
        spec: Frozen preprocessing contract.
    """
    if args.platform not in spec["platforms"]:
        raise ValueError("Platform is outside this variant's frozen matrix")
    manifest, images = load_dataset(args.manifest, args.images_root,
                                    expected_count=args.expected_images, labeled=False)
    evaluation = json.loads(args.evaluation_manifest.read_text())
    if len(evaluation["images"]) != evaluation["count"]:
        raise ValueError("Evaluation manifest count mismatch")
    overlap = {r["sha256"] for _, r in images} & {r["sha256"] for r in evaluation["images"]}
    if overlap:
        raise ValueError("Calibration/evaluation image hash overlap")
    args.output.mkdir(parents=True, exist_ok=False)
    data_dir = args.output / "data"
    data_dir.mkdir()
    outputs = []
    for index, (path, record) in enumerate(images):
        data = calibration_input(prepare_rgb(path, spec["contract"]), spec["contract"], args.platform)
        # X5 OE 1.2.8 uses fromfile even for .npy names; S OE 3.7.0 supports npy.
        output = data_dir / (f"{index:06d}.rgb" if args.platform == "x5" else f"{index:06d}.npy")
        if args.platform == "x5":
            data.astype("<f4", copy=False).tofile(output)
        else:
            np.save(output, data)
        outputs.append({"file": output.name, "sha256": sha256_file(output),
                        "source_sha256": record["sha256"], "shape": list(data.shape),
                        "min": float(data.min()), "max": float(data.max())})
    result = receipt("calibration-inputs", args.model, spec)
    result.update(status="prepared", platform=args.platform, count=len(images),
                  dataset=manifest["dataset"], manifest_sha256=sha256_file(args.manifest),
                  evaluation_manifest_sha256=sha256_file(args.evaluation_manifest),
                  overlap_count=0, outputs=outputs,
                  domain="rgb_0_255" if args.platform == "x5" else "normalized_onnx_input",
                  storage="raw_little_endian_float32" if args.platform == "x5" else "npy_float32",
                  suitability="candidate; quantify accuracy loss in P2")
    write_json(args.output / "calibration.json", result)


def make_config(args: argparse.Namespace, spec: dict) -> None:
    """Generate an OE configuration bound to verified graph/calibration bytes.

    Args:
        args: Export directory, prepared calibration, platform, output.
        spec: Frozen input/normalization contract.
    """
    import yaml

    graph, export = verify_export(args.export_dir, args.model, spec)
    calibration = json.loads((args.calibration_dir / "calibration.json").read_text())
    if (calibration["spec"] != spec or calibration["model"] != args.model
            or calibration["platform"] != args.platform or calibration["status"] != "prepared"
            or args.platform not in spec["platforms"]):
        raise ValueError("Calibration/model/platform contract mismatch")
    expected_files = {item["file"] for item in calibration["outputs"]}
    actual_files = {path.name for path in (args.calibration_dir / "data").iterdir()}
    if not expected_files or expected_files != actual_files:
        raise ValueError("Calibration directory has missing or extra files")
    for item in calibration["outputs"]:
        if sha256_file(args.calibration_dir / "data" / item["file"]) != item["sha256"]:
            raise ValueError("Calibration bytes changed")
    contract = spec["contract"]
    size = contract["size"]
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = args.model.replace("-", "_") + "_" + args.platform
    config = {
        "model_parameters": {"onnx_model": str(graph.resolve()), "march": MARCH[args.platform],
                             "working_dir": str((args.output / "build").resolve()),
                             "output_model_file_prefix": prefix},
        "input_parameters": {"input_name": "data", "input_type_rt": "nv12",
                             "input_space_and_range": "bt601_video",
                             "input_type_train": "rgb", "input_layout_train": "NCHW",
                             "input_shape": f"1x3x{size}x{size}", "norm_type": "data_mean_and_scale",
                             "mean_value": " ".join(format(x * 255, ".12g") for x in contract["mean"]),
                             "scale_value": " ".join(format(1 / (x * 255), ".12g") for x in contract["std"])},
        "calibration_parameters": {"cal_data_dir": str((args.calibration_dir / "data").resolve()),
                                   "cal_data_type": "float32", "calibration_type": "default"},
        "compiler_parameters": {"optimize_level": "O3" if args.platform == "x5" else "O2"},
    }
    if args.platform == "x5":
        config["calibration_parameters"]["preprocess_on"] = False
        config["compiler_parameters"]["compile_mode"] = "latency"
        config["compiler_parameters"]["core_num"] = 1
        config["compiler_parameters"]["input_source"] = {"data": "pyramid"}
    config_path = args.output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    result = receipt("conversion-config", args.model, spec)
    result.update(status="prepared-not-compiled", platform=args.platform, march=MARCH[args.platform],
                  config_sha256=sha256_file(config_path), onnx_sha256=export["onnx_sha256"],
                  calibration_receipt_sha256=sha256_file(args.calibration_dir / "calibration.json"),
                  calibration_batch="toolchain default; actual batch must be read from compile log",
                  command_inside_container=("hb_mapper makertbin --model-type onnx --config "
                                            if args.platform == "x5" else "hb_compile --config ") + str(config_path.resolve()),
                  mount_requirement="Mount input/output host paths at the same absolute container paths")
    write_json(args.output / "config.json", result)


def evaluate(args: argparse.Namespace, spec: dict) -> None:
    """Evaluate every labeled image and retain predictions plus exact identity.

    Args:
        args: Export, dataset, provider, and new output directory.
        spec: Frozen model input/output contract.
    """
    graph, export = verify_export(args.export_dir, args.model, spec)
    manifest, images = load_dataset(args.manifest, args.images_root,
                                    expected_count=args.expected_images)
    session = make_session(graph, spec["contract"], args.provider, args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path = args.output / "predictions.jsonl"
    correct1 = correct5 = 0
    start = time.monotonic()
    with prediction_path.open("x") as output:
        for index, (path, record) in enumerate(images, 1):
            data = float_input(prepare_rgb(path, spec["contract"]), spec["contract"])
            scores = session.run(None, {session.get_inputs()[0].name: data})[0]
            ids = top5_ids(scores)
            label = record["label_id"]
            correct1 += int(ids[0] == label)
            correct5 += int(label in ids)
            output.write(json.dumps({"image": record["path"], "sha256": record["sha256"],
                                     "label_id": label, "top5": ids,
                                     "top5_logits": scores.reshape(-1)[ids].tolist()}) + "\n")
            if index % 500 == 0:
                print(f"{index}/{len(images)} top1={correct1/index:.5f} top5={correct5/index:.5f}", flush=True)
    result = receipt("float-onnx-evaluation", args.model, spec)
    result.update(status="passed", evaluation_scope="full_manifest",
                  dataset=manifest["dataset"], image_count=len(images),
                  class_count=1000, manifest_sha256=sha256_file(args.manifest),
                  onnx_sha256=export["onnx_sha256"],
                  export_receipt_sha256=sha256_file(args.export_dir / "export.json"),
                  predictions_sha256=sha256_file(prediction_path),
                  metrics={"top1": correct1 / len(images), "top5": correct5 / len(images)},
                  correct={"top1": correct1, "top5": correct5},
                  ort_providers=session.get_providers(), requested_provider=args.provider,
                  wall_seconds=time.monotonic() - start,
                  timing_scope="diagnostic host wall time; not board performance",
                  baseline="FP32 ONNX, RGB crop; no NV12 roundtrip", board_status="not-run")
    write_json(args.output / "evaluation.json", result)


def main(default_family: str | None = None, default_command: str | None = None) -> None:
    """Parse the shared CLI, optionally restricted by a sample entrypoint.

    Args:
        default_family: Allow only this sample's model variants when set.
        default_command: Sample entrypoint's fixed subcommand when set.
    """
    pins = json.loads(PINS.read_text())["models"]
    models = [key for key, value in pins.items() if default_family is None or value["family"] == default_family]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "export", "calibrate", "config", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--model", choices=models, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "export":
            command.add_argument("--source-dir", type=Path, required=True)
            command.add_argument("--images", type=Path, nargs="+", required=True)
            command.add_argument("--opset", type=int, choices=(11, 19), default=11)
            command.add_argument("--simplify", action="store_true")
        if name in ("export", "evaluate"):
            command.add_argument("--threads", type=int, default=4)
        if name in ("calibrate", "evaluate"):
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--images-root", type=Path, required=True)
            command.add_argument("--expected-images", type=int, required=True)
        if name in ("config", "evaluate"):
            command.add_argument("--export-dir", type=Path, required=True)
        if name in ("calibrate", "config"):
            command.add_argument("--platform", choices=tuple(MARCH), required=True)
        if name == "calibrate":
            command.add_argument("--evaluation-manifest", type=Path, required=True)
        if name == "config":
            command.add_argument("--calibration-dir", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--provider", choices=("CPUExecutionProvider", "CUDAExecutionProvider"),
                                 default="CPUExecutionProvider")
    argv = sys.argv[1:]
    if default_command:
        argv = [default_command, *argv]
    args = parser.parse_args(argv)
    spec = pins[args.model]
    validate_contract(spec["contract"])
    actions = {"fetch": fetch_source, "export": export_model, "calibrate": prepare_calibration,
               "config": make_config, "evaluate": evaluate}
    actions[args.command](args, spec)


if __name__ == "__main__":
    main()
