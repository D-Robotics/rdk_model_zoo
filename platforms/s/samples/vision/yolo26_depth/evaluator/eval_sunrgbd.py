"""Evaluate YOLO26 Depth outputs on a prepared SUN RGB-D subset."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for an evaluation artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_bilinear(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a depth map with PyTorch bilinear interpolation.

    Args:
        depth: Source depth map.
        height: Target height.
        width: Target width.

    Returns:
        Resized float32 depth map.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))[None, None]
    return F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()


CALIBRATION = {"n": (1.0, -0.19384765625), "s": (1.0, 1.71484375), "m": (1.0, 1.630859375), "l": (1.0, -0.2498779296875), "x": (1.0, -0.316650390625)}


def postprocess_raw(
    raw_logit: np.ndarray,
    record: dict,
    protocol: str,
    size: int,
    variant: str,
) -> np.ndarray:
    """Decode a lite raw logit according to the deployment protocol.

    Args:
        raw_logit: Raw HBM/ONNX output before the lite external postprocess.
        record: Dataset record containing source geometry.
        protocol: Evaluation preprocessing protocol.
        size: Square model input size.
        variant: YOLO26 model scale used to resolve calibration constants.

    Returns:
        Postprocessed relative-depth map.
    """
    raw_logit = np.asarray(raw_logit, dtype=np.float32).squeeze()
    if protocol != "deployment_scale_fill":
        raise ValueError(protocol)
    cal_a, cal_b = CALIBRATION[variant]
    depth = np.exp(np.clip(raw_logit, -4.0, 5.0) * cal_a + cal_b).astype(np.float32)
    original_height, original_width = record["original_hw"]
    return resize_bilinear(depth, original_height, original_width)


def prepare_ground_truth(
    source_root: Path,
    record: dict,
    protocol: str,
    size: int,
) -> np.ndarray:
    """Prepare ground truth according to the evaluation protocol.

    Args:
        source_root: Root directory containing prepared depth arrays.
        record: Dataset record containing geometry metadata.
        protocol: Evaluation preprocessing protocol.
        size: Square model input size.

    Returns:
        Ground-truth depth map aligned with the prediction protocol.
    """
    depth_path = source_root / record["depth_m"]
    depth = np.load(depth_path).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if tuple(depth.shape[:2]) != tuple(record["original_hw"]):
        raise ValueError(f"GT shape mismatch for {depth_path}: {depth.shape} != {record['original_hw']}")
    if protocol != "deployment_scale_fill":
        raise ValueError(protocol)
    return depth


class DepthMetrics:
    """Accumulate standard monocular-depth metrics.

    Args:
        align: Prediction alignment mode used before metric calculation.
        min_depth: Minimum valid target depth.
        max_depth: Maximum valid target depth.

    Attributes:
        align: Selected alignment mode.
        totals: Accumulated metric totals.
        count: Number of evaluated images.
        scales: Per-image alignment scales.
    """
    def __init__(self, align: str, min_depth: float = 0.001, max_depth: float = 100.0) -> None:
        self.align = align
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.totals = torch.zeros(7, dtype=torch.float64)
        self.count = 0.0
        self.scales = []

    def update(self, prediction: np.ndarray, target: np.ndarray) -> dict:
        """Align one prediction and append its depth metrics.

        Args:
            prediction: Predicted relative-depth map.
            target: Ground-truth metric-depth map.

        Returns:
            Metrics calculated for the current image.
        """
        prediction_tensor = torch.from_numpy(np.ascontiguousarray(prediction, dtype=np.float32))
        target_tensor = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
        mask = (
            (target_tensor > self.min_depth)
            & (target_tensor < self.max_depth)
            & torch.isfinite(prediction_tensor)
        )
        count = int(mask.sum())
        if count == 0:
            return {"valid_pixels": 0, "scale": None}
        prediction_valid = prediction_tensor[mask].float()
        target_valid = target_tensor[mask].float()
        scale = 1.0
        if self.align == "median":
            scale_tensor = torch.median(target_valid) / torch.median(
                prediction_valid.clamp_min(self.min_depth)
            )
            scale = float(scale_tensor)
            prediction_valid = prediction_valid * scale_tensor
        prediction_valid = prediction_valid.clamp(self.min_depth, self.max_depth)
        threshold = torch.maximum(
            prediction_valid / target_valid,
            target_valid / prediction_valid,
        )
        log_diff = torch.log(prediction_valid) - torch.log(target_valid)
        totals = torch.stack(
            [
                (threshold < 1.25).sum(),
                (threshold < 1.25**2).sum(),
                (threshold < 1.25**3).sum(),
                (torch.abs(prediction_valid - target_valid) / target_valid).sum(),
                ((prediction_valid - target_valid) ** 2).sum(),
                (log_diff**2).sum(),
                log_diff.sum(),
            ]
        )
        self.totals += totals.cpu().double()
        self.count += float(count)
        self.scales.append(scale)
        single = DepthMetrics(self.align, self.min_depth, self.max_depth)
        single.totals = totals.cpu().double()
        single.count = float(count)
        single.scales = [scale]
        result = single.result()
        result.update({"valid_pixels": count, "scale": scale})
        return result

    def result(self) -> dict:
        """Average all accumulated per-image metrics.

        Returns:
            Aggregate metric dictionary.
        """
        if self.count == 0:
            return {
                "delta1": 0.0,
                "delta2": 0.0,
                "delta3": 0.0,
                "abs_rel": 0.0,
                "rmse": 0.0,
                "silog": 0.0,
                "valid_pixels": 0,
            }
        delta1, delta2, delta3, abs_rel, rmse_sq, silog_a, silog_b = (
            float(value) for value in self.totals
        )
        count = self.count
        silog = max((silog_a / count) - (silog_b / count) ** 2, 0.0) ** 0.5 * 100
        result = {
            "delta1": delta1 / count,
            "delta2": delta2 / count,
            "delta3": delta3 / count,
            "abs_rel": abs_rel / count,
            "rmse": math.sqrt(rmse_sq / count),
            "silog": silog,
            "valid_pixels": int(count),
        }
        if self.scales:
            result["median_scale_mean"] = float(np.mean(self.scales))
            result["median_scale_std"] = float(np.std(self.scales))
        return result


def fidelity_metrics(candidate: list[np.ndarray], reference: list[np.ndarray]) -> dict:
    """Compare two ordered sets of model outputs.

    Args:
        candidate: Candidate output arrays.
        reference: Reference output arrays.

    Returns:
        Aggregate numerical fidelity metrics.
    """
    candidate_all = np.concatenate([array.astype(np.float64).ravel() for array in candidate])
    reference_all = np.concatenate([array.astype(np.float64).ravel() for array in reference])
    valid = np.isfinite(candidate_all) & np.isfinite(reference_all)
    candidate_valid = candidate_all[valid]
    reference_valid = reference_all[valid]
    delta = candidate_valid - reference_valid
    denominator = np.linalg.norm(candidate_valid) * np.linalg.norm(reference_valid)
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "max_abs": float(np.abs(delta).max()),
        "mean_relative_abs": float(
            (np.abs(delta) / np.maximum(np.abs(reference_valid), 1e-6)).mean()
        ),
        "cosine_similarity": float(np.dot(candidate_valid, reference_valid) / denominator)
        if denominator
        else 0.0,
        "finite_values": int(valid.sum()),
    }


def load_output(path: Path, key: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Load ordered output arrays and associated metadata.

    Args:
        path: JSON or NumPy output artifact.
        key: Output field to extract from structured artifacts.

    Returns:
        Record identifiers and output arrays keyed by identifier.
    """
    archive = np.load(path)
    if key not in archive:
        raise KeyError(f"{key} is not present in {path}")
    indices = archive["indices"].astype(np.int32)
    values = archive[key].astype(np.float32)
    return indices, {int(index): values[position] for position, index in enumerate(indices)}


def main() -> None:
    """Parse arguments and evaluate a prepared SUN RGB-D subset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--float-outputs", required=True, type=Path)
    parser.add_argument("--quant-outputs", required=True, type=Path)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument(
        "--protocol",
        choices=("deployment_scale_fill",),
        required=True,
    )
    parser.add_argument("--variant", choices=("n", "s", "m", "l", "x"), required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(args.report)
    prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    size = int(prepared["size"])
    records = {int(record["index"]): record for record in prepared["records"]}
    float_indices, float_logs = load_output(args.float_outputs, "float_raw")
    quant_indices, quant_logs = load_output(args.quant_outputs, "quant_raw")
    if set(quant_indices.tolist()) - set(float_indices.tolist()):
        raise ValueError("quant outputs contain indices missing from float outputs")
    selected_indices = sorted(int(index) for index in quant_indices)

    accumulators = {
        "float_none": DepthMetrics("none"),
        "float_median": DepthMetrics("median"),
        "quant_none": DepthMetrics("none"),
        "quant_median": DepthMetrics("median"),
    }
    float_depths = []
    quant_depths = []
    float_log_values = []
    quant_log_values = []
    per_sample = []
    for index in selected_indices:
        record = records[index]
        target = prepare_ground_truth(args.source_root, record, args.protocol, size)
        float_depth = postprocess_raw(float_logs[index], record, args.protocol, size, args.variant)
        quant_depth = postprocess_raw(quant_logs[index], record, args.protocol, size, args.variant)
        float_none = accumulators["float_none"].update(float_depth, target)
        float_median = accumulators["float_median"].update(float_depth, target)
        quant_none = accumulators["quant_none"].update(quant_depth, target)
        quant_median = accumulators["quant_median"].update(quant_depth, target)
        per_sample.append(
            {
                "index": index,
                "sample": record["sample"],
                "sensor": record["sensor"],
                "float_none": float_none,
                "float_median": float_median,
                "quant_none": quant_none,
                "quant_median": quant_median,
            }
        )
        float_depths.append(float_depth)
        quant_depths.append(quant_depth)
        float_log_values.append(float_logs[index])
        quant_log_values.append(quant_logs[index])

    float_metrics = {
        "none": accumulators["float_none"].result(),
        "median": accumulators["float_median"].result(),
    }
    quant_metrics = {
        "none": accumulators["quant_none"].result(),
        "median": accumulators["quant_median"].result(),
    }
    report = {
        "schema_version": "1.0",
        "candidate": args.candidate_name,
        "protocol": args.protocol,
        "sample_count": len(selected_indices),
        "indices": selected_indices,
        "metric_contract": {
            "min_depth": 0.001,
            "max_depth": 100.0,
            "aggregation": "pixel-pooled across validation set",
            "median_alignment": "per-image torch.median(gt) / torch.median(pred.clamp_min(0.001))",
            "prediction_clamp": [0.001, 100.0],
        },
        "float_vs_gt": float_metrics,
        "quant_vs_gt": quant_metrics,
        "accuracy_delta_quant_minus_float": {
            align: {
                key: quant_metrics[align][key] - float_metrics[align][key]
                for key in ("delta1", "delta2", "delta3", "abs_rel", "rmse", "silog")
            }
            for align in ("none", "median")
        },
        "quant_vs_float": {
            "raw_logit": fidelity_metrics(quant_log_values, float_log_values),
            "postprocessed_depth": fidelity_metrics(quant_depths, float_depths),
        },
        "prepared_manifest": str(args.prepared_manifest),
        "prepared_manifest_sha256": sha256(args.prepared_manifest),
        "float_outputs": str(args.float_outputs),
        "float_outputs_sha256": sha256(args.float_outputs),
        "quant_outputs": str(args.quant_outputs),
        "quant_outputs_sha256": sha256(args.quant_outputs),
        "per_sample": per_sample,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "per_sample"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
