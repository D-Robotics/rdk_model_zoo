"""Capture OSNet features on X5/S-series or FP32 ONNX, or score an existing capture."""

import argparse
import json
import platform
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from samples.vision.osnet.evaluator.metrics import records, retrieval
from samples.vision.osnet.model.artifacts import load_manifest, sha256


def dataset_sha256_manifest(dataset, ordered):
    return [{"path": str(path.relative_to(dataset)), "sha256": sha256(path)} for path, _, _ in ordered]


def require_matching_capture(capture_dir, manifest, dataset_manifest):
    """Accept cached features only when names and image bytes still match."""
    saved = json.loads((Path(capture_dir) / "manifest.json").read_text())
    if saved != manifest:
        raise ValueError("Capture manifest differs from the sorted evaluation images")
    dataset_path = Path(capture_dir) / "dataset-sha256.json"
    if not dataset_path.is_file():
        raise ValueError("Capture is missing dataset-sha256.json")
    if json.loads(dataset_path.read_text()) != dataset_manifest:
        raise ValueError("Capture dataset-sha256 differs from the current Market1501 image hashes")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Market1501 root")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--capture-dir", type=Path, default=None, help="Previously saved features.npy, manifest.json and dataset-sha256.json; skips inference")
    parser.add_argument("--backend", choices=("x5", "s100", "s100p", "s600", "onnx"), default="x5", help="Feature capture backend")
    parser.add_argument("--model-path", type=Path, default=None, help="Exact BIN/HBM or spatial FP32 ONNX")
    parser.add_argument("--variant", choices=tuple(load_manifest()["artifacts"]), default="spatial-rgb-late8")
    parser.add_argument("--raw", action="store_true", help="Score unnormalized features")
    parser.add_argument("--capture-only", action="store_true", help="Save full features for scoring on the host")
    parser.add_argument("--onnx-threads", type=int, default=4, help="CPU ONNX inference threads")
    args = parser.parse_args(argv)
    if args.onnx_threads < 1:
        parser.error("--onnx-threads must be positive")
    started = time.time()
    if args.output_dir.exists():
        parser.error("--output-dir must be a new directory")
    import numpy as np
    from PIL import Image

    query, gallery = records(args.dataset / "query"), records(args.dataset / "bounding_box_test")
    if (len(query), len(gallery)) != (3368, 15913):
        parser.error("Full evaluation requires 3368 query and 15913 valid gallery images")
    ordered = query + gallery
    manifest = [{"name": path.name, "pid": pid, "camera": camera} for path, pid, camera in ordered]
    provenance = {"board_test": "not-run", "python": platform.python_version(), "numpy": np.__version__}
    dataset_manifest = None
    if args.capture_dir:
        dataset_manifest = dataset_sha256_manifest(args.dataset, ordered)
        require_matching_capture(args.capture_dir, manifest, dataset_manifest)
        features = np.load(args.capture_dir / "features.npy", allow_pickle=False)
        provenance.update(capture_dir=str(args.capture_dir), features_sha256=sha256(args.capture_dir / "features.npy"))
    else:
        if args.model_path is None:
            parser.error("--model-path is required for feature capture")
        provenance.update(backend=args.backend, model_sha256=sha256(args.model_path))
        if args.backend != "onnx":
            from samples.vision.osnet.runtime.python.matching import OSNet, OSNetConfig
            from samples.vision.osnet.model.artifacts import require_artifact
            require_artifact(args.model_path, args.variant)
            model = OSNet(OSNetConfig(args.model_path, target=args.backend, normalize=False))
            extract = model.predict
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.onnx_threads
            session = ort.InferenceSession(str(args.model_path), options, providers=["CPUExecutionProvider"])
            provenance.update(onnxruntime=ort.__version__, providers=session.get_providers(), onnx_threads=args.onnx_threads)

            def extract(image):
                x = np.asarray(image.convert("RGB").resize((128, 256), Image.Resampling.BILINEAR), np.float32).transpose(2, 0, 1) / np.float32(255)
                x = ((x - np.array([.485, .456, .406], np.float32)[:, None, None]) /
                     np.array([.229, .224, .225], np.float32)[:, None, None])[None]
                return session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(x)})[0].reshape(512)
        features = []
        for index, (path, _, _) in enumerate(ordered):
            with Image.open(path) as image:
                features.append(extract(image))
            if (index + 1) % 500 == 0:
                print(f"Captured {index + 1}/{len(ordered)}", flush=True)
        features = np.stack(features)
        if args.backend != "onnx":
            provenance["board_test"] = "executed"
    if not np.isfinite(features).all() or features.shape != (len(ordered), 512):
        raise ValueError("Capture must have finite 512-dimensional features for the full dataset")
    result = ({"status": "captured", "samples": len(ordered), "query": len(query), "gallery": len(gallery)} if args.capture_only else
              retrieval(query, gallery, features[:len(query)], features[len(query):], normalize=not args.raw))
    result["provenance"] = provenance
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "features.npy", features)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance["elapsed_seconds"] = time.time() - started
    provenance["features_sha256"] = sha256(args.output_dir / "features.npy")
    if dataset_manifest is None:
        dataset_manifest = dataset_sha256_manifest(args.dataset, ordered)
    (args.output_dir / "dataset-sha256.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n")
    provenance["dataset_manifest_sha256"] = sha256(args.output_dir / "dataset-sha256.json")
    (args.output_dir / ("capture.json" if args.capture_only else "accuracy.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
