# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export trained torchvision ViT/FCOS graphs and freeze real calibration inputs.

This creates source ONNX graphs, not board models. FCOS keeps all three dense
prediction tensors; its CPU box decoding and NMS remain outside the graph.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json

SOURCE_COMMIT = "824e8c8726b65fd9d5abdc9702f81c2b0c4c0dc8"
ATOL, RTOL = 1e-4, 1e-3


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def build_source(task, checkpoint):
    import torch
    import torchvision
    if torchvision.__version__.split("+")[0] != "0.23.0":
        raise ValueError("This exporter binds torchvision 0.23.0")
    if task == "vit":
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=None)
    else:
        from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
        weights = FCOS_ResNet50_FPN_Weights.COCO_V1
        model = fcos_resnet50_fpn(weights=None, weights_backbone=None,
                                 min_size=512, max_size=512,
                                 score_thresh=0.2, nms_thresh=0.6)
    prefix = weights.url.rsplit("-", 1)[-1].split(".")[0]
    if not identity(checkpoint)["sha256"].startswith(prefix):
        raise ValueError("Checkpoint differs from the official publisher hash prefix")
    loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Incomplete trained state")
    return model.eval(), weights


def preprocess(task, image):
    """Return the explicit F32 source input, with no hidden NV12 preprocessing."""
    import numpy as np
    from PIL import Image
    if task == "vit":
        from torchvision.models import ViT_B_16_Weights
        with Image.open(image) as frame:
            return ViT_B_16_Weights.IMAGENET1K_V1.transforms()(frame.convert("RGB"))[None].numpy()
    import cv2
    bgr = cv2.imread(str(image))
    if bgr is None:
        raise ValueError("Unreadable calibration image: " + str(image))
    rgb = cv2.cvtColor(cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    tensor = (tensor - np.asarray([.485, .456, .406], dtype=np.float32)[None, :, None, None])
    return np.ascontiguousarray(tensor / np.asarray([.229, .224, .225], dtype=np.float32)[None, :, None, None])


def export(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import torchvision
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model, weights = build_source(args.task, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    cal = args.output / "calibration"
    cal.mkdir()
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(images) != args.count or args.count < 100:
        raise ValueError("Provide the complete frozen representative image subset")
    records = []
    for index, path in enumerate(images):
        value = preprocess(args.task, path)
        out = cal / f"{index:04d}.npy"
        np.save(out, value, allow_pickle=False)
        records.append({"image": identity(path), "tensor": identity(out), "shape": list(value.shape), "dtype": str(value.dtype)})
    manifest = {"count": len(records), "task": args.task, "source_commit": SOURCE_COMMIT,
                "calibration_domain": "exact source ONNX input; normalized float32 NCHW NPY",
                "geometry": "PIL bilinear resize-shorter 256 and center crop 224" if args.task == "vit" else "OpenCV INTER_LINEAR direct stretch 512x512",
                "normalization": {"scale": 1 / 255, "mean": [.485, .456, .406], "std": [.229, .224, .225]},
                "producer": identity(__file__), "files": records}
    (args.output / "calibration.manifest.json").write_text(json.dumps(manifest, indent=2))
    indices = [0, len(records) // 2, len(records) - 1]
    cases = [(f"real-{i}", np.load(records[i]["tensor"]["path"], allow_pickle=False)) for i in indices]
    shape = tuple(cases[0][1].shape)
    cases.append(("random-seed0", np.random.default_rng(0).normal(0, 1, shape).astype(np.float32)))
    if args.task == "vit":
        names = ["logits"]
        # Exportable explicit attention retains the entire classifier output.
        torch.backends.mha.set_fastpath_enabled(False)
        wrapper = model
        def reference(value):
            return (model(value),)
    else:
        names = ["cls_logits", "bbox_regression", "bbox_ctrness"]
        class DenseFCOS(torch.nn.Module):
            def __init__(self, source):
                super().__init__()
                self.backbone, self.head = source.backbone, source.head
            def forward(self, value):
                heads = self.head(list(self.backbone(value).values()))
                return tuple(heads[name] for name in names)
        wrapper = DenseFCOS(model).eval()
        def reference(value):
            heads = model.head(list(model.backbone(value).values()))
            return tuple(heads[name] for name in names)
    path = args.output / "model.onnx"
    references = []
    with torch.inference_mode():
        for _, value in cases:
            source = reference(torch.from_numpy(value))
            actual = wrapper(torch.from_numpy(value))
            actual = (actual,) if isinstance(actual, torch.Tensor) else actual
            for candidate, expected in zip(actual, source, strict=True):
                np.testing.assert_array_equal(candidate.numpy(), expected.numpy())
            references.append([t.numpy() for t in source])
        torch.onnx.export(wrapper, torch.from_numpy(cases[0][1]), str(path), input_names=["images"],
                          output_names=names, opset_version=19, do_constant_folding=True, dynamo=False)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    checks = []
    for (name, value), expected in zip(cases, references, strict=True):
        outputs = session.run(None, {"images": value})
        comparisons = []
        for output_name, candidate, target in zip(names, outputs, expected, strict=True):
            if candidate.shape != target.shape or candidate.dtype != target.dtype or not np.isfinite(candidate).all():
                raise ValueError("Wrong output contract: " + output_name)
            violations = int(np.count_nonzero(~np.isclose(candidate, target, atol=ATOL, rtol=RTOL)))
            comparisons.append({"name": output_name, "shape": list(candidate.shape), "dtype": str(candidate.dtype),
                                "max_abs": float(np.max(np.abs(candidate - target))), "violations": violations,
                                "passed": violations == 0})
        checks.append({"input": name, "comparisons": comparisons, "status": "passed" if all(x["passed"] for x in comparisons) else "failed"})
    report = {"schema_version": 1, "status": "passed" if all(x["status"] == "passed" for x in checks) else "failed",
              "source": {"repository": "https://github.com/pytorch/vision", "commit": SOURCE_COMMIT,
                         "version": "v0.23.0", "weights": str(weights), "weight_url": weights.url,
                         "checkpoint": identity(args.checkpoint), "license": "BSD-3-Clause"},
              "native_commit": args.native_commit, "onnx": identity(path), "exporter": identity(__file__),
              "source_implementation": identity(inspect.getfile(type(model))), "calibration_manifest": identity(args.output / "calibration.manifest.json"),
              "parameter_count": sum(p.numel() for p in model.parameters()), "input": {"name": "images", "shape": list(shape), "dtype": "float32"},
              "outputs": [{"name": n, "shape": list(v.shape), "dtype": str(v.dtype)} for n, v in zip(names, references[0], strict=True)],
              "categories": weights.meta["categories"], "normalization": manifest["normalization"], "geometry": manifest["geometry"],
              "atol": ATOL, "rtol": RTOL, "raw_wrapper_bit_exact": True, "cases": checks,
              "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
              "board": "not-run", "dataset_accuracy": "not-run"}
    (args.output / "export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "task": args.task, "cases": checks}), flush=True)
    return 0 if report["status"] == "passed" else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("vit", "fcos"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--native-commit", required=True)
    return export(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
