"""Export the pinned official MODNet and prepare native-runtime calibration.

Run with a separately prepared official checkout/checkpoint. Outputs go to a
new directory; the source checkout, weights and calibration images stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT = "28165a451e4610c9d77cfdf925a94610bb2810fb"
CHECKPOINT_URL = "https://drive.usercontent.google.com/download?id=1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz&export=download&confirm=t"
PREFIX = "modnet_photographic_portrait_512x512_rgb_bayese_20261102"


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    calibration = args.calibration_dir.resolve()
    output = args.output.resolve()
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT:
        raise ValueError("MODNet checkout must match the pinned upstream commit")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("MODNet source checkout must be clean")
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    manifest = json.loads((calibration / "manifest.json").read_text())
    if manifest["count"] != 200 or len(manifest["images"]) != 200 or manifest["seed"] != 42:
        raise ValueError("Expected the fixed 200-image seed-42 calibration manifest")
    output.mkdir(parents=True, exist_ok=False)
    (output / "calibration").mkdir()

    # Import installed onnx before adding upstream: upstream also has onnx/.
    import onnx
    import onnxruntime as ort
    import torch
    import numpy as np
    import cv2
    import yaml
    torch.set_num_threads(2)
    torch.manual_seed(42)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from samples.vision.modnet.runtime.python.modnet import preprocess_image
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("official_modnet_onnx", source / "onnx/modnet_onnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.MODNet(backbone_pretrained=False).cpu().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    from src.models.modnet import MODNet
    native = MODNet(backbone_pretrained=False).cpu().eval()
    native.load_state_dict(state, strict=True)
    graph = output / (PREFIX + ".onnx")
    dummy = torch.rand(1, 3, 512, 512) * 2 - 1
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(graph), input_names=["input"], output_names=["matte"],
                          opset_version=11, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(graph))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 512, 512]
    assert session.get_outputs()[0].shape == [1, 1, 512, 512]
    samples, checks = [], []
    for index, row in enumerate(manifest["images"]):
        image_path = calibration / row["name"]
        if sha256(image_path) != row["sha256"]:
            raise ValueError(f"Calibration source changed: {image_path}")
        tensor, context = preprocess_image(cv2.imread(str(image_path)))
        assert tensor.dtype == np.float32 and tensor.shape == (1, 3, 512, 512)
        assert np.isfinite(tensor).all()
        target = output / "calibration" / (image_path.stem + ".bin")
        tensor.tofile(target)
        samples.append({"source": str(image_path), "source_sha256": row["sha256"],
                        "tensor": str(target), "sha256": sha256(target), "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype), "minimum": float(tensor.min()),
                        "maximum": float(tensor.max()), "mean": float(tensor.mean()), "nonfinite": 0})
        if index in (0, 51, 143):
            reread = np.fromfile(target, dtype=np.float32).reshape(1, 3, 512, 512)
            np.testing.assert_array_equal(tensor, reread)
            with torch.no_grad():
                reference = native(torch.from_numpy(reread), True)[2].numpy()
            actual = session.run(None, {"input": reread})[0]
            assert np.isfinite(reference).all() and np.isfinite(actual).all()
            np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
            checks.append({"input": row["name"], "input_sha256": sha256(target),
                           "max_abs_error": float(np.max(np.abs(actual-reference))),
                           "finite": True, "shape": list(actual.shape)})
    with torch.no_grad():
        reference = native(dummy, True)[2].numpy()
    actual = session.run(None, {"input": dummy.numpy()})[0]
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-4)
    checks.append({"input": "seed42-uniform-minus1-plus1", "max_abs_error": float(np.max(np.abs(actual-reference))), "finite": True})
    config = {
        "model_parameters": {"onnx_model": str(graph), "march": "bayes-e", "layer_out_dump": False,
                             "working_dir": str(output.parent / "output"), "output_model_file_prefix": PREFIX},
        "input_parameters": {"input_name": "input", "input_type_rt": "featuremap", "input_type_train": "featuremap",
                             "input_layout_train": "NCHW", "input_layout_rt": "NCHW", "input_shape": "1x3x512x512", "norm_type": "no_preprocess"},
        "calibration_parameters": {"cal_data_dir": str(output / "calibration"), "cal_data_type": "float32", "calibration_type": "default"},
        "compiler_parameters": {"compile_mode": "latency", "core_num": 1, "input_source": {"input": "ddr"},
                                "debug": False, "optimize_level": "O3", "jobs": 2}}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "calibration-manifest.json").write_text(json.dumps({"count": 200, "seed": 42, "source_manifest_sha256": sha256(calibration / "manifest.json"), "images": samples}, indent=2) + "\n")
    report = {"source_commit": SOURCE_COMMIT, "checkpoint_url": CHECKPOINT_URL, "checkpoint_sha256": sha256(checkpoint),
              "checkpoint_version": "sha256:" + sha256(checkpoint), "model_license": "Apache-2.0", "opset": 11,
              "onnx": str(graph), "onnx_sha256": sha256(graph), "config_sha256": sha256(output / "config.yaml"),
              "exporter_sha256": sha256(__file__), "official_model_code_sha256": sha256(source / "onnx/modnet_onnx.py"),
              "preprocess_code_sha256": sha256(Path(__file__).resolve().parents[1] / "runtime/python/modnet.py"),
              "torch": torch.__version__, "onnx_version": onnx.__version__, "onnxruntime": ort.__version__,
              "checks": checks, "atol": 1e-4, "rtol": 1e-3, "passed": True,
              "preprocessing": "BGR->RGB; (pixel-127.5)/127.5; long-side INTER_AREA to 512; center zero-pad; float32 NCHW"}
    (output / "prepare-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "preprocess-report.md").write_text(report["preprocessing"] + "\n200 COCO person images, seed42; all source digests verified.\nThree calibration tensors reread and compared against PyTorch/ORT, plus one deterministic random input.\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
