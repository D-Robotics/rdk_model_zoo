#!/usr/bin/env python3
"""Snapshot the real YOLO26 task-b8 S-platform receipts into isolated drafts.

This script only writes under model_zoo_web/candidate-staging/.
It never uploads artifacts or modifies the active catalog/release inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml


WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEB_ROOT.parent
WORKBENCH = Path(os.environ.get("RDK_MODEL_ZOO_WORKBENCH", REPO_ROOT.parent / "rdk_model_zoo_workbench"))
EXPERIMENT = WORKBENCH / "experiments" / "yolo26-tasks-b8"
CANDIDATE_ROOT = WEB_ROOT / "candidate-staging"
STAGING: Path | None = None
TASKS = ("cls", "seg", "pose", "obb")
SIZES = ("n", "s", "m", "l", "x")
PLATFORMS = ("s600", "s100p", "s100")
METRIC_MAPS = {
    "cls": {"top1": "top1", "top5": "top5"},
    "seg": {"bbox/AP": "box_ap", "segm/AP": "mask_ap"},
    "pose": {"keypoints/AP": "keypoints_ap"},
    "obb": {"mAP50": "map_50"},
}
TASK_TITLES = {
    "cls": ("图像分类", "Image classification"),
    "seg": ("实例分割", "Instance segmentation"),
    "pose": ("姿态估计", "Pose estimation"),
    "obb": ("旋转框检测", "Oriented object detection"),
}
DATASET_LABELS = {
    "cls": "ImageNetV2 MatchedFrequency (not ILSVRC2012 val)",
    "seg": "COCO val2017",
    "pose": "COCO val2017",
    "obb": "DOTA val",
}
RECEIPT_DATASET_IDS = {
    "cls": "ImageNetV2 matched-frequency full 10000",
    "seg": "COCO val2017 instance segmentation (5000)",
    "pose": "COCO val2017 person keypoints (5000)",
    "obb": "DOTA-v1.0 val single-scale (458)",
}
DATASET_TASKS = {
    "cls": "classification",
    "seg": "instance segmentation",
    "pose": "person keypoint estimation",
    "obb": "oriented object detection",
}
REQUIRED_AUDIT_CHECKS = (
    "batch8_conversion",
    "board_receipt_identity",
    "board_full_dataset",
    "board_result_log_and_driver_binding",
    "build_receipt_hash",
    "benchmark_build_receipt_copy",
    "model_onnx_checkpoint_hashes",
    "oe_report_hashes_and_model_binding",
    "float_baseline_and_conversion_onnx_binding",
    "valid_float_board_comparison",
    "runtime_performance_1_and_2_threads",
    "runtime_perf_input_and_model_binding",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve_snapshot_output(candidate_root: Path, supplied: str | Path) -> tuple[Path, str]:
    raw_path = Path(supplied).expanduser()
    raw_path = raw_path if raw_path.is_absolute() else candidate_root / raw_path
    require(not raw_path.is_symlink(), "snapshot output may not be a symlink")
    snapshot_path = raw_path.resolve()
    try:
        relative_path = snapshot_path.relative_to(candidate_root.resolve())
    except ValueError as exc:
        raise ValueError("snapshot output must stay below candidate-staging") from exc
    require(len(relative_path.parts) == 1, "snapshot output must be a direct candidate-staging child")
    snapshot_id = snapshot_path.name
    require(re.fullmatch(r"yolo26-task-b8-audit-[0-9]{8}T[0-9]{6}Z(?:-r[1-9][0-9]*)?", snapshot_id) is not None,
            "snapshot id must look like yolo26-task-b8-audit-YYYYMMDDTHHMMSSZ[-rN]")
    return snapshot_path, snapshot_id


def candidate_metrics(task: str, receipt_metrics: dict[str, Any]) -> dict[str, float]:
    mapped: dict[str, float] = {}
    for receipt_field, web_field in METRIC_MAPS[task].items():
        value = receipt_metrics.get(receipt_field)
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1,
            f"{task}: receipt metric {receipt_field} must be a finite ratio",
        )
        mapped[web_field] = float(value)
    return mapped


def stage_cover_assets(staging: Path) -> None:
    source_root = WEB_ROOT / "release" / "assets"
    target_root = staging / "release" / "assets"
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_source = source_root / "yolo26-task-banners.json"
    require(manifest_source.is_file(), f"cover manifest is missing: {manifest_source}")
    manifest = read_json(manifest_source)
    for task in TASKS:
        manifest_item = manifest["items"][task]
        source = source_root / manifest_item["file"]
        require(source.is_file(), f"task cover is missing: {source}")
        require(source.stat().st_size == manifest_item["size_bytes"], f"cover size mismatch: {source}")
        require(sha256(source) == manifest_item["sha256"], f"cover SHA mismatch: {source}")
        shutil.copy2(source, target_root / source.name)
    shutil.copy2(manifest_source, target_root / manifest_source.name)


def require_regular(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            f"{label} is missing, empty, or a symlink: {path}")


def require_audit_check(entry: dict[str, Any], name: str, label: str) -> None:
    checks = {row.get("name"): row for row in entry.get("checks", [])}
    row = checks.get(name)
    require(isinstance(row, dict) and row.get("passed") is True,
            f"{label}: authoritative audit check {name!r} did not pass")


def audit_driver_log_path(entry: dict[str, Any], board_receipt: dict[str, Any],
                          benchmark: Path, label: str) -> Path:
    check = next((row for row in entry.get("checks", [])
                  if row.get("name") == "board_result_log_and_driver_binding"), None)
    require(isinstance(check, dict) and check.get("passed") is True,
            f"{label}: board/driver binding audit is missing")
    match = re.search(r"driver log:\s*([0-9a-f]{64})", check.get("detail", ""))
    require(match is not None, f"{label}: audit does not identify a driver log SHA")
    expected_sha = match.group(1)
    evaluation = board_receipt.get("evaluation", {})
    explicit = evaluation.get("driver_log")
    candidates = [Path(explicit)] if isinstance(explicit, str) else []
    candidates.extend((benchmark / "output" / "board_eval_driver.log",
                       benchmark / "output" / "board_result.json.board.log"))
    matches = [path for path in dict.fromkeys(candidates)
               if path.is_file() and not path.is_symlink() and sha256(path) == expected_sha]
    require(len(matches) == 1,
            f"{label}: driver log SHA from audit must bind exactly one local benchmark log")
    return matches[0]


def comparison_metrics(task: str, receipt: dict[str, Any], board_metrics: dict[str, Any],
                       float_metrics: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    mapping = METRIC_MAPS[task]
    raw = receipt.get("metrics")
    require(isinstance(raw, dict) and set(raw) == set(mapping),
            f"{task}: comparator metric keys do not match the Web schema")
    board_web = candidate_metrics(task, board_metrics)
    float_web: dict[str, float] = {}
    comparison_web: dict[str, Any] = {}
    for receipt_field, web_field in mapping.items():
        source = raw[receipt_field]
        require(isinstance(source, dict), f"{task}: comparator row {receipt_field} is not an object")
        fv, bv, delta, retention = (source.get("float"), source.get("board"),
                                    source.get("board_minus_float"), source.get("retention_ratio"))
        values = (fv, bv, delta, retention)
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(float(value)) for value in values),
                f"{task}: comparator row {receipt_field} contains a non-finite value")
        require(0 <= fv <= 1 and 0 <= bv <= 1,
                f"{task}: comparator accuracy value {receipt_field} is outside [0, 1]")
        require(bv == board_metrics[receipt_field] and bv == source.get("board"),
                f"{task}: authoritative board metric differs from comparator for {receipt_field}")
        require(fv == float_metrics.get(receipt_field),
                f"{task}: comparator float metric differs from float receipt for {receipt_field}")
        require(math.isclose(delta, bv - fv, rel_tol=0, abs_tol=1e-12),
                f"{task}: comparator delta does not recompute for {receipt_field}")
        expected_retention = bv / fv if fv else 0.0
        require(math.isclose(retention, expected_retention, rel_tol=0, abs_tol=1e-12),
                f"{task}: comparator retention does not recompute for {receipt_field}")
        float_web[web_field] = float(fv)
        comparison_web[web_field] = {
            "float": float(fv),
            "board": float(bv),
            "board_minus_float": float(delta),
            "retention_ratio": float(retention),
        }
    return float_web, board_web, comparison_web


def end_to_end_record(task: str, source: dict[str, Any]) -> dict[str, Any]:
    require(source.get("schema_version") == 1, f"{task}: C++ E2E schema version mismatch")
    require(source.get("pipeline_streams") in (1, 2), f"{task}: invalid C++ E2E stream count")
    require(source.get("runtime_submission_threads") == source.get("pipeline_streams"),
            f"{task}: C++ E2E stream/thread count mismatch")
    require(isinstance(source.get("metrics_ms"), dict) and source.get("throughput_fps", 0) > 0,
            f"{task}: C++ E2E metrics are incomplete")
    return {
        "tool": f"ultralytics_yolo_{task}",
        "schema_version": source["schema_version"],
        "implementation": source["implementation"],
        "timing_scope": source["timing_scope"],
        "pipeline_streams": source["pipeline_streams"],
        "runtime_submission_threads": source["runtime_submission_threads"],
        "cpu_thread_policy": source.get("cpu_thread_policy"),
        "online_cpu_threads": source.get("online_cpu_threads"),
        "opencv_threads": source.get("opencv_threads"),
        "warmup_frames_per_round": source.get("warmup_frames_per_round"),
        "runs_per_round": source.get("runs_per_round"),
        "frames_per_round": source.get("frames_per_stream_per_round"),
        "rounds": source.get("rounds"),
        "timed_frames": source.get("timed_frames"),
        "aggregate_wall_ms": source.get("aggregate_wall_ms"),
        "metrics_ms": source["metrics_ms"],
        "throughput_fps": source["throughput_fps"],
        "output_kind": source.get("output_kind"),
    }


def verify_and_stage_entry(entry: dict[str, Any], audit: dict[str, Any], audit_sha: str,
                           config_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
                           report_dir: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task, size, platform = entry.get("task"), entry.get("size"), entry.get("platform")
    label = entry.get("name")
    require((task, size, platform) in {
        (t, s, p) for t in TASKS for s in SIZES for p in PLATFORMS
    }, f"authoritative audit has unsupported identity {label!r}")
    require(label == f"{task}-{size}-{platform}", f"authoritative audit identity/name mismatch: {label}")
    for check in REQUIRED_AUDIT_CHECKS:
        require_audit_check(entry, check, label)

    benchmark = Path(entry["benchmark"])
    require(benchmark.resolve().is_relative_to((WORKBENCH / "runs" / "benchmark").resolve()),
            f"{label}: benchmark path is outside the Workbench benchmark tree")
    output = benchmark / "output"
    board_receipt_path = Path(entry["board_receipt"])
    require(board_receipt_path == output / "board_result.json.receipt.json",
            f"{label}: authoritative audit board receipt path is unexpected")
    board_result_path = output / "board_result.json"
    comparison_path = output / "comparison_receipt.json"
    performance_path = output / "performance.json"
    performance_input_path = benchmark / "input" / "perf-input.json"
    board_model_sha_path = benchmark / "logs" / "board-model.sha256"
    for path, name in (
        (board_receipt_path, "board receipt"), (board_result_path, "board result"),
        (comparison_path, "comparison receipt"), (performance_path, "performance evidence"),
        (performance_input_path, "performance input manifest"),
        (board_model_sha_path, "board HBM SHA sidecar"),
    ):
        require_regular(path, f"{label} {name}")

    board_receipt = read_json(board_receipt_path)
    comparison = read_json(comparison_path)
    performance_source = read_json(performance_path)
    performance_input = read_json(performance_input_path)
    board_result_sha = sha256(board_result_path)
    board_receipt_sha = sha256(board_receipt_path)
    comparison_sha = sha256(comparison_path)
    performance_sha = sha256(performance_path)
    performance_input_sha = sha256(performance_input_path)

    require(board_receipt.get("status") == "metrics-computed", f"{label}: board receipt is incomplete")
    require(board_receipt.get("model", {}).get("task") == task and
            board_receipt.get("model", {}).get("size") == size and
            board_receipt.get("model", {}).get("platform") == platform,
            f"{label}: board receipt identity mismatch")
    require(board_receipt.get("model", {}).get("board_sha256_verified") is True,
            f"{label}: board receipt did not verify its HBM SHA")
    board_metrics = board_receipt.get("evaluation", {}).get("metrics")
    require(isinstance(board_metrics, dict) and board_metrics == entry.get("board_metrics"),
            f"{label}: current audit and board receipt metrics differ")
    require(board_receipt.get("evaluation", {}).get("result_sha256") == board_result_sha,
            f"{label}: board result hash does not match its receipt")
    require(board_receipt.get("dataset", {}).get("full_split") is True and
            board_receipt["dataset"].get("counted_images") == board_receipt["dataset"].get("expected_images"),
            f"{label}: board receipt does not cover the complete validation split")

    build_receipt_path = Path(board_receipt["conversion"]["receipt_path"])
    build_receipt_sha = board_receipt["conversion"]["receipt_sha256"]
    require_regular(build_receipt_path, f"{label} build receipt")
    require(sha256(build_receipt_path) == build_receipt_sha,
            f"{label}: build receipt hash differs from board receipt")
    build_receipt = read_json(build_receipt_path)
    require(build_receipt.get("status") == "host-compiled", f"{label}: build receipt is not host compiled")
    require(build_receipt.get("model", {}).get("task") == task and
            build_receipt.get("model", {}).get("size") == size and
            build_receipt.get("target", {}).get("platform") == platform,
            f"{label}: build receipt identity mismatch")
    artifact = build_receipt["artifacts"]["model"]
    artifact_sha = artifact.get("sha256")
    require(board_receipt["model"].get("sha256") == artifact_sha,
            f"{label}: board HBM SHA does not match build receipt")
    local_artifact_path = build_receipt_path.parent / artifact["name"]
    require_regular(local_artifact_path, f"{label} local HBM")
    require(local_artifact_path.stat().st_size == artifact["size_bytes"] and
            sha256(local_artifact_path) == artifact_sha,
            f"{label}: local HBM size/SHA differs from build receipt")
    target_onnx = build_receipt.get("artifacts", {}).get("onnx", {})
    target_onnx_path = build_receipt_path.parent / target_onnx.get("name", "")
    require_regular(target_onnx_path, f"{label} target ONNX")
    target_onnx_sha = sha256(target_onnx_path)
    require(target_onnx_sha == target_onnx.get("sha256") and
            target_onnx_sha == board_receipt.get("conversion", {}).get("onnx_sha256"),
            f"{label}: target ONNX SHA differs from build/board receipts")

    receipt_campaign = board_receipt.get("campaign", {})
    campaign_path = Path(receipt_campaign["path"])
    require_regular(campaign_path, f"{label} campaign")
    campaign_sha = sha256(campaign_path)
    require(campaign_sha == receipt_campaign.get("sha256") and
            campaign_sha == comparison.get("campaign", {}).get("sha256"),
            f"{label}: board/comparator campaign SHA mismatch")
    require(comparison.get("model", {}).get("task") == task and
            comparison["model"].get("size") == size and
            comparison["model"].get("platform") == platform and
            comparison["model"].get("compiled_model_sha256") == artifact_sha and
            comparison["model"].get("board_target_onnx_sha256") ==
            board_receipt.get("conversion", {}).get("onnx_sha256"),
            f"{label}: comparison model/HBM identity mismatch")
    require(comparison.get("status") == "comparison-computed" and
            comparison.get("direct_metric_comparison") == "valid" and
            comparison.get("retention_gate_eligible") is True and
            comparison.get("comparison_scope", {}).get("comparable") is True,
            f"{label}: current comparator is not valid and retention eligible")
    compare_inputs = comparison.get("inputs", {})
    require(compare_inputs.get("board_receipt", {}).get("path") == str(board_receipt_path) and
            compare_inputs.get("board_receipt", {}).get("sha256") == board_receipt_sha,
            f"{label}: comparator is not bound to the current board receipt")
    require(compare_inputs.get("board_result_sha256") == board_result_sha,
            f"{label}: comparator is not bound to the current board result")

    float_ref = compare_inputs.get("float_result")
    require(isinstance(float_ref, dict) and isinstance(float_ref.get("path"), str),
            f"{label}: comparator has no bound float result")
    float_result_path = Path(float_ref["path"])
    expected_float_output = (WORKBENCH / "runs" / "benchmark" /
                             f"yolo26-{task}-{size}-s600-releaseperf" / "output")
    require(float_result_path.parent == expected_float_output and
            float_result_path.name in {"float_result.json", "float_result_reference.json"},
            f"{label}: comparator float result is not the frozen S600 task/size reference")
    require_regular(float_result_path, f"{label} float result")
    float_result_sha = sha256(float_result_path)
    require(float_ref.get("sha256") == float_result_sha,
            f"{label}: comparator float-result SHA mismatch")
    float_result = read_json(float_result_path)
    require(float_result.get("status") == "metrics-computed" and
            float_result.get("model", {}).get("task") == task and
            float_result["model"].get("size") == size,
            f"{label}: float result task/size identity mismatch")
    float_build_path = Path(float_result["conversion"]["receipt_path"])
    require_regular(float_build_path, f"{label} float-reference build receipt")
    float_build_sha = sha256(float_build_path)
    require(float_result["conversion"].get("receipt_sha256") == float_build_sha,
            f"{label}: float result build receipt SHA mismatch")
    float_build = read_json(float_build_path)
    float_artifact = float_build.get("artifacts", {}).get("model", {})
    float_onnx = float_build.get("artifacts", {}).get("onnx", {})
    require(float_result.get("conversion", {}).get("compiled_model_sha256") == float_artifact.get("sha256") and
            float_result.get("model", {}).get("compiled_model_sha256") == float_artifact.get("sha256") and
            float_result.get("conversion", {}).get("onnx_sha256") == float_onnx.get("sha256") and
            float_result.get("model", {}).get("float_onnx_sha256") == float_onnx.get("sha256") and
            float_result.get("model", {}).get("checkpoint_sha256") ==
            build_receipt.get("provenance", {}).get("source_sha256"),
            f"{label}: float-reference HBM/checkpoint/ONNX hashes are not bound")
    float_onnx_path = Path(float_result["model"]["float_onnx"])
    require_regular(float_onnx_path, f"{label} float ONNX")
    require(sha256(float_onnx_path) == float_onnx.get("sha256") and
            comparison.get("model", {}).get("float_onnx_sha256") == float_onnx.get("sha256"),
            f"{label}: comparison float ONNX SHA is not bound to the float result")
    float_predictions_path = float_result_path.parent / "float_predictions.json"
    require_regular(float_predictions_path, f"{label} float predictions")
    require(compare_inputs.get("float_predictions_sha256") ==
            sha256(float_predictions_path),
            f"{label}: comparator is not bound to current float predictions")
    float_metrics = float_result.get("evaluation", {}).get("metrics")
    require(isinstance(float_metrics, dict), f"{label}: float result metrics are missing")
    float_web, board_web, comparison_web = comparison_metrics(
        task, comparison, board_metrics, float_metrics,
    )
    count = board_receipt["dataset"]["counted_images"]
    require(comparison.get("dataset", {}).get("board_counted_images") == count and
            comparison["dataset"].get("float_evaluated_images") == count and
            comparison["dataset"].get("expected_images") == count,
            f"{label}: comparison is not bound to the same complete dataset")

    width, height = build_receipt["conversion"]["geometry"]["size"]
    config_key = (task, size)
    if config_key not in config_cache:
        config_path = WORKBENCH / "configs" / f"campaign-yolo26-{task}-{size}-v2.json"
        require_regular(config_path, f"{label} frozen campaign")
        config = read_json(config_path)
        config_cache[config_key] = (config_path, config)
    config_path, config = config_cache[config_key]
    require(config.get("task") == task and config.get("size") == size and
            config.get("geometry", {}).get("size") == [width, height],
            f"{label}: model campaign geometry mismatch")
    require(build_receipt.get("conversion", {}).get("input_shape") == [1, 3, height, width],
            f"{label}: build input shape differs from campaign")
    require(performance_source.get("tool") == "hrt_model_exec perf" and
            performance_source.get("implementation") == "native_cpp_cli" and
            performance_source.get("timing_scope") == "model_runtime" and
            performance_source.get("thread_semantics") == "runtime_submission_concurrency" and
            performance_source.get("input_kind") == "fixed_nv12" and
            performance_source.get("fixed_input") == performance_input and
            performance_source.get("fixed_input", {}).get("model_size") == [width, height],
            f"{label}: runtime performance evidence/input contract mismatch")
    require(performance_source.get("stages", {}).get("runtime") == "measured" and
            len(performance_source.get("measurements", [])) == 2 and
            {m.get("threads") for m in performance_source["measurements"]} == {1, 2} and
            len(performance_source.get("raw_measurements", [])) == 6,
            f"{label}: runtime performance does not contain both conditions and six runs")
    board_model_sha = board_model_sha_path.read_text(encoding="utf-8").split()[0]
    require(board_model_sha == artifact_sha,
            f"{label}: runtime performance board-model SHA differs from HBM")
    performance_planes: list[dict[str, Any]] = []
    plane_evidence: list[dict[str, Any]] = []
    for plane in performance_input.get("planes", []):
        plane_path = benchmark / "input" / plane["file"]
        require_regular(plane_path, f"{label} performance plane {plane.get('name')}")
        plane_sha = sha256(plane_path)
        require(plane_sha == plane.get("sha256") and
                plane_path.stat().st_size == plane.get("size_bytes"),
                f"{label}: performance input plane hash/size mismatch")
        performance_planes.append({
            key: plane[key] for key in ("name", "shape", "size_bytes", "sha256")
        })
        plane_evidence.append({"path": str(plane_path), "sha256": plane_sha})
    require(len(performance_planes) == 2,
            f"{label}: expected both NV12 input planes")

    performance: dict[str, Any] = {
        "status": "measured",
        "tool": performance_source["tool"],
        "implementation": performance_source["implementation"],
        "timing_scope": performance_source["timing_scope"],
        "thread_semantics": performance_source["thread_semantics"],
        "core_id": performance_source["core_id"],
        "warmup_frames_per_condition": performance_source["warmup_frames_per_condition"],
        "warmup_scope": performance_source.get("warmup_scope"),
        "runs_per_condition": performance_source["runs_per_condition"],
        "frames_per_run": performance_source["frames_per_run"],
        "input_kind": performance_source["input_kind"],
        "stages": performance_source["stages"],
        "measurements": performance_source["measurements"],
        "raw_measurements": performance_source["raw_measurements"],
        "fixed_input": {
            "source_image_sha256": performance_input.get("source_image_sha256"),
            "preprocessing": performance_input.get("preprocessing"),
            "resize_type": performance_input.get("resize_type"),
            "model_size": [width, height],
            "padding": performance_input.get("padding"),
            "planes": performance_planes,
        },
        "end_to_end": [],
        "end_to_end_status": "pending",
    }
    e2e_paths: list[dict[str, str]] = []
    cpp_ok = entry.get("checks", [])
    cpp_check_map = {row.get("name"): row.get("passed") is True for row in cpp_ok}
    if cpp_check_map.get("cpp_end_to_end_1_and_2_streams") is True:
        for check in (
            "cpp_model_sha_and_alias_binding", "cpp_run_release_receipt_provenance",
            "cpp_runtime_source_sha", "cpp_executable_sha", "cpp_e2e_logs",
        ):
            require_audit_check(entry, check, label)
        e2e_provenance_path = output / "e2e_provenance.json"
        require_regular(e2e_provenance_path, f"{label} C++ E2E provenance")
        e2e_provenance = read_json(e2e_provenance_path)
        require(e2e_provenance.get("task") == task and e2e_provenance.get("size") == size and
                e2e_provenance.get("platform") == platform and
                e2e_provenance.get("model", {}).get("sha256") == artifact_sha and
                e2e_provenance.get("board_eval_receipt", {}).get("sha256") == board_receipt_sha and
                e2e_provenance.get("conversion_receipt", {}).get("sha256") == build_receipt_sha,
                f"{label}: C++ E2E provenance does not bind the board/build model")
        for streams in (1, 2):
            e2e_path = output / f"e2e-{streams}.json"
            require_regular(e2e_path, f"{label} C++ E2E {streams}-stream receipt")
            e2e_data = read_json(e2e_path)
            require(e2e_data.get("pipeline_streams") == streams and
                    e2e_data.get("runtime_submission_threads") == streams and
                    e2e_data.get("model", "").endswith(artifact["name"]) and
                    e2e_data.get("runtime_source_sha256") == e2e_provenance.get("runtime_source_sha256") and
                    e2e_data.get("executable_sha256") == e2e_provenance.get("execution", {}).get("binary_sha256"),
                    f"{label}: C++ E2E {streams}-stream identity mismatch")
            performance["end_to_end"].append(end_to_end_record(task, e2e_data))
            e2e_paths.append({"path": str(e2e_path), "sha256": sha256(e2e_path)})
        performance["end_to_end_status"] = "evidence-present-pending-release-review"
        e2e_paths.append({"path": str(e2e_provenance_path), "sha256": sha256(e2e_provenance_path)})
    else:
        require(cpp_check_map.get("cpp_end_to_end_1_and_2_streams") is False,
                f"{label}: C++ E2E audit state is missing")

    report_source = build_receipt_path.parent / "oe_report_data.json"
    require_regular(report_source, f"{label} structured OE report data")
    oe_data = read_json(report_source)
    require(oe_data.get("provenance", {}).get("artifact_sha256") == artifact_sha,
            f"{label}: OE JSON does not bind the current HBM")
    report_name = f"{task}-{size}-{platform}-oe-data.json"
    report_target = report_dir / report_name
    shutil.copy2(report_source, report_target)
    require(sha256(report_target) == sha256(report_source), f"{label}: copied OE JSON changed")

    accuracy: dict[str, Any] = {
        "dataset": "DOTA val" if task == "obb" else DATASET_LABELS[task],
        "receipt_dataset": board_receipt["dataset"]["id"],
        "task": DATASET_TASKS[task],
        "images": count,
        "board_runtime": board_web,
        "float_onnx": float_web,
        "runtime": board_web,
        "comparison": {
            "status": "valid-evidence",
            "direct_metric_comparison": "valid",
            "retention_gate_eligible": True,
            "metrics": comparison_web,
            "comparison_scope": comparison["comparison_scope"],
            "limitations": comparison.get("limitations", []),
            "human_accuracy_delta_review": "pending",
        },
        "human_review": {"status": "pending", "reviewer": None, "receipt": None},
    }
    if task == "obb":
        accuracy["evaluation_scope"] = "local_dota_val_single_scale"
        accuracy["scope_note"] = board_receipt["evaluation"].get("dota_score", {}).get("scope_note")

    model_id = f"ultralytics_yolo/yolo26/{task}"
    release_id = f"{model_id}/{size}/{platform}"
    board_driver_path = audit_driver_log_path(entry, board_receipt, benchmark, label)
    runtime_manifest_ref = compare_inputs.get("runtime_source_manifest")
    require(isinstance(runtime_manifest_ref, dict) and isinstance(runtime_manifest_ref.get("path"), str),
            f"{label}: runtime source manifest reference is missing")
    runtime_manifest_path = Path(runtime_manifest_ref["path"])
    require_regular(runtime_manifest_path, f"{label} runtime source manifest")
    runtime_manifest_sha = sha256(runtime_manifest_path)
    require(runtime_manifest_sha == runtime_manifest_ref.get("sha256"),
            f"{label}: runtime source manifest hash mismatch")
    build_evidence = {
        "path": str(build_receipt_path), "sha256": build_receipt_sha,
        "campaign_path": str(campaign_path), "campaign_sha256": campaign_sha,
        "artifact_path": str(local_artifact_path), "artifact_sha256": artifact_sha,
    }
    evidence = {
        "id": release_id,
        "task": task,
        "size": size,
        "platform": platform,
        "authoritative_audit": {"path": str(audit_path), "sha256": audit_sha,
                                "generated_at": audit["generated_at"]},
        "board_receipt": {"path": str(board_receipt_path), "sha256": board_receipt_sha},
        "board_result": {"path": str(board_result_path), "sha256": board_result_sha},
        "board_driver_log": {"path": str(board_driver_path), "sha256": sha256(board_driver_path)},
        "build": build_evidence,
        "target_onnx": {"path": str(target_onnx_path), "sha256": target_onnx_sha},
        "comparison_receipt": {"path": str(comparison_path), "sha256": comparison_sha},
        "float_result": {"path": str(float_result_path), "sha256": float_result_sha},
        "float_reference_build": {"path": str(float_build_path), "sha256": float_build_sha,
                                   "artifact_sha256": float_artifact.get("sha256"),
                                   "onnx_path": str(float_onnx_path), "onnx_sha256": float_onnx.get("sha256")},
        "float_predictions": {"path": str(float_predictions_path),
                              "sha256": compare_inputs["float_predictions_sha256"]},
        "performance": {"path": str(performance_path), "sha256": performance_sha,
                        "input_path": str(performance_input_path), "input_sha256": performance_input_sha,
                        "board_model_sha_path": str(board_model_sha_path),
                        "board_model_sha256": board_model_sha,
                        "plane_files": plane_evidence},
        "cpp_end_to_end": {"status": performance["end_to_end_status"], "files": e2e_paths},
        "oe_data": f"release/reports/{report_name}",
        "oe_data_sha256": sha256(report_target),
        "board_metrics": board_metrics,
        "float_metrics": {key: value for key, value in zip(METRIC_MAPS[task], float_web.values())},
        "web_metrics": board_web,
        "comparison_metrics": comparison_web,
        "direct_metric_comparison": "valid",
        "retention_gate_eligible": True,
        "human_accuracy_delta_review": "pending",
        "human_release_approval": "pending",
        "artifact": artifact,
    }

    config_path_cached, config_cached = config_cache[config_key]
    width, height = config_cached["geometry"]["size"]
    platform_record = {
        "platform": platform,
        "status": "candidate",
        "artifact": {
            "filename": artifact["name"],
            "format": build_receipt["target"]["format"],
            "march": build_receipt["target"]["march"],
            "runtime_input": build_receipt["conversion"]["runtime_input"],
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact_sha,
            "local_status": "verified",
            "local_path": str(local_artifact_path),
            "oss_url": None,
        },
        "reports": {"oe_data": f"release/reports/{report_name}", "oe_conversion_url": None},
        "accuracy": accuracy,
        "performance": performance,
        "provenance": {
            "build_id": build_receipt["build_id"],
            "campaign_build_id": receipt_campaign["build_id"],
            "source_checkpoint_sha256": build_receipt.get("provenance", {}).get("source_sha256"),
            "source_weight_url": config_cached.get("source_url"),
            "remote_source_sha256": None,
            "workbench_repository_commit": build_receipt.get("provenance", {}).get("repository_commit"),
            "board_sha256_verified": True,
            "board_runtime_source_manifest": {
                "path": str(runtime_manifest_path),
                "sha256": runtime_manifest_sha,
            },
            "direct_metric_comparison": "valid",
            "retention_gate_eligible": True,
            "human_accuracy_delta_review": "pending",
            "human_release_approval": None,
        },
        "release_gates": {
            "oss_artifact_upload": "pending",
            "oss_release_manifest": "pending",
            "oss_checksums": "pending",
            "oss_oe_report": "pending",
            "float_accuracy_comparator": "valid-evidence-pending-human-review",
            "remote_source_sha256": "pending",
            "runtime_performance": "measured-pending-release-review",
            "end_to_end_performance": performance["end_to_end_status"],
            "human_accuracy_delta_review": "pending",
            "human_release_approval": "pending",
            "release_status": "pending",
        },
        "human_approval": None,
        "evidence": evidence,
    }
    require(platform_record["artifact"]["oss_url"] is None and
            platform_record["reports"]["oe_conversion_url"] is None and
            platform_record["human_approval"] is None,
            f"{label}: candidate row unexpectedly contains upload or approval data")
    return platform_record, evidence, config_cached


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def build_snapshot(staging: Path, snapshot_id: str, audit: dict[str, Any], audit_sha: str,
                   audit_path: Path, review_path: Path, review_sha: str) -> int:
    require(EXPERIMENT.is_dir(), f"workbench experiment is missing: {EXPERIMENT}")
    require(audit.get("scope", {}).get("expected") == 60, "current audit scope must be exactly 60")
    require(audit.get("ready") == 0, "expected current audit to remain release blocked")
    entries = audit.get("entries")
    require(isinstance(entries, list) and len(entries) == 60, "current audit must have exactly 60 entries")
    expected = {(task, size, platform) for task in TASKS for size in SIZES for platform in PLATFORMS}
    observed = [(e.get("task"), e.get("size"), e.get("platform")) for e in entries]
    require(len(set(observed)) == 60 and set(observed) == expected,
            "current audit must contain 60 unique task/size/platform variants")
    for entry in entries:
        gate_checks = {row.get("name"): row for row in entry.get("checks", []) if isinstance(row, dict)}
        require(gate_checks.get("human_accuracy_delta_review", {}).get("passed") is False and
                gate_checks.get("human_release_approval", {}).get("passed") is False,
                f"{entry.get('name')}: human accuracy and release gates must remain pending")
    review_doc = read_json(review_path)
    review_rows = review_doc.get("rows") or review_doc.get("entries")
    require(isinstance(review_rows, list) and len(review_rows) == len(entries),
            "review matrix must cover the same 60 audit rows")
    audit_report_ref = review_doc.get("audit_report") or {}
    require(audit_report_ref.get("path") == str(audit_path)
            and audit_report_ref.get("sha256") == audit_sha,
            "review matrix does not bind the selected authoritative audit path/SHA")
    review_keys = [(row.get("task"), row.get("size"), row.get("platform")) for row in review_rows]
    audit_keys = [(row.get("task"), row.get("size"), row.get("platform")) for row in entries]
    require(len(set(review_keys)) == len(review_keys) and set(review_keys) == set(audit_keys),
            "review matrix identities do not exactly match the authoritative audit")
    for row in review_rows:
        review_technical = row.get("technical_audit", {})
        require(review_technical.get("audit_report_path") == str(audit_path)
                and review_technical.get("audit_report_sha256") == audit_sha,
                f"{row.get('variant')}: review row does not bind the selected audit path/SHA")

    stage_cover_assets(staging)
    report_dir = staging / "release" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    data_root = staging / "data" / "vision" / "ultralytics_yolo" / "yolo26"
    descriptions = {task: {"zh": TASK_TITLES[task][0], "en": TASK_TITLES[task][1]} for task in TASKS}
    task_records: dict[str, dict[str, Any]] = {}
    input_models: dict[str, Any] = {}
    input_releases: dict[str, Any] = {}
    config_cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}
    evidence_rows: list[dict[str, Any]] = []
    cpp_count = 0

    for task in TASKS:
        model_id = f"ultralytics_yolo/yolo26/{task}"
        input_models[model_id] = {
            "cover": f"assets/yolo26-{task}.webp",
            "cover_label": "AI 生成的任务示意图，非模型推理结果",
        }
        task_records[task] = {
            "schema_version": 1,
            "id": model_id,
            "name": f"YOLO26 {task.upper()}",
            "domain": "vision",
            "source": "ultralytics_yolo",
            "provider": "ultralytics",
            "family": "yolo26",
            "task": task,
            "status": "candidate",
            "release_status": "pending",
            "description": descriptions[task],
            "license": {"name": "AGPL-3.0", "url": "https://www.ultralytics.com/license"},
            "sample_path": "samples/vision/ultralytics_yolo",
            "cover": {
                "status": "schematic_candidate",
                "source": f"release/assets/yolo26-{task}.webp",
                "manifest": "release/assets/yolo26-task-banners.json",
                "sha256": sha256(WEB_ROOT / "release" / "assets" / f"yolo26-{task}.webp"),
                "claim": "AI-generated task illustration; not a model inference result",
            },
            "dataset": {
                "name": DATASET_LABELS[task],
                "receipt_name": RECEIPT_DATASET_IDS[task],
                "task": DATASET_TASKS[task],
                "images": 10000 if task == "cls" else 458 if task == "obb" else 5000,
                "scope": "local_dota_val_single_scale" if task == "obb" else "full_validation_split",
                "official_test_claim": False if task == "obb" else None,
            },
            "release_gates": {
                "oss_artifact_upload": "pending",
                "oss_release_manifest": "pending",
                "oss_checksums": "pending",
                "oss_oe_report": "pending",
                "float_accuracy_comparator": "valid-evidence-pending-human-review",
                "remote_source_sha256": "pending",
                "runtime_performance": "measured-pending-release-review",
                "end_to_end_performance": "partially-available-pending-review",
                "human_accuracy_delta_review": "pending",
                "human_release_approval": "pending",
                "release_status": "pending",
            },
            "variants": [],
        }

    for task in TASKS:
        record = task_records[task]
        for size in SIZES:
            config_path = WORKBENCH / "configs" / f"campaign-yolo26-{task}-{size}-v2.json"
            require_regular(config_path, f"{task}-{size} campaign")
            config = read_json(config_path)
            require(config.get("task") == task and config.get("size") == size,
                    f"campaign identity mismatch: {config_path}")
            width, height = config["geometry"]["size"]
            variant = {
                "size": size,
                "model": {
                    "parameter_count": config["model"]["parameter_count"],
                    "gflops": config["model"]["gflops"],
                    "gflops_method": config["model"].get("measurement_method"),
                },
                "input": {
                    "width": width, "height": height,
                    "shape": [1, 3, height, width],
                    "geometry_source": "authoritative campaign and build receipt",
                    "source_preprocess_contract": "pending independent review",
                },
                "platforms": [],
            }
            for platform in PLATFORMS:
                entry = next(e for e in entries if e.get("name") == f"{task}-{size}-{platform}")
                platform_record, evidence, _ = verify_and_stage_entry(
                    entry, audit, audit_sha, config_cache, report_dir, audit_path,
                )
                variant["platforms"].append(platform_record)
                release_id = evidence["id"]
                input_releases[release_id] = {"oe_data": f"reports/{Path(evidence['oe_data']).name}"}
                evidence_rows.append(evidence)
                cpp_count += bool(evidence["cpp_end_to_end"]["files"])
            record["variants"].append(variant)

    require(len(evidence_rows) == 60 and len({row["id"] for row in evidence_rows}) == 60,
            "staging did not produce exactly 60 unique evidence rows")
    require(isinstance(cpp_count, int) and 0 <= cpp_count <= len(evidence_rows),
            "staging C++ E2E count is inconsistent")
    for task, record in task_records.items():
        (data_root / f"{task}.yaml").parent.mkdir(parents=True, exist_ok=True)
        (data_root / f"{task}.yaml").write_text(
            yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8",
        )

    release_root = staging / "release"
    write_json(release_root / "inputs.json", {
        "schema_version": 1, "models": input_models, "releases": input_releases,
    })
    manifest_path = release_root / "assets" / "yolo26-task-banners.json"
    write_json(staging / "evidence-index.json", {
        "schema_version": 2,
        "snapshot_id": snapshot_id,
        "record_count": len(evidence_rows),
        "snapshot_status": "candidate-pending",
        "authoritative_audit": {"path": str(audit_path), "sha256": audit_sha,
                                "generated_at": audit["generated_at"], "ready": audit["ready"]},
        "review_matrix": {"path": str(review_path), "sha256": review_sha},
        "summary": {
            "batch8_conversion": 60,
            "board_full_dataset": 60,
            "valid_float_board_comparison": 60,
            "runtime_performance_1_and_2_threads": 60,
            "cpp_end_to_end_1_and_2_streams": cpp_count,
            "human_accuracy_delta_review_pending": 60,
            "human_release_approval_pending": 60,
        },
        "cover_manifest": "release/assets/yolo26-task-banners.json",
        "cover_manifest_sha256": sha256(manifest_path),
        "records": evidence_rows,
    })
    readme = f"""# YOLO26 Cls/Seg/Pose/OBB S-platform candidate snapshot

This immutable offline snapshot was built from authoritative Workbench audit `{audit_path}`
generated at `{audit['generated_at']}` (SHA-256 `{audit_sha}`). It contains four task YAML drafts and
exactly 60 unique size/platform rows. All 60 rows have verified build/board HBM identity, full-split
board results, valid float-vs-board comparator records, and 1/2-thread fixed-NV12 runtime performance.

The comparator and performance fields are machine-verified evidence. Human accuracy-delta review and
release approval are still pending for all 60 rows, so all records remain `status: candidate` and
`release_status: pending`; the snapshot contains no approvals, OSS artifact/report URLs, or uploaded
release objects. C++ application-style end-to-end receipts are present for {cpp_count}/60 rows.
Runtime FPS below is HRT `model_runtime` on a fixed NV12 input; it excludes preprocessing, postprocessing,
and application/camera end-to-end time.

The four cover images are AI-generated schematic task illustrations, not model inference results.
OBB precision is local DOTA-v1.0 val, single-scale, 458 images; it is not an official DOTA test score.
The review matrix `{review_path}` is bound to audit SHA-256 `{audit_sha}` and has SHA-256 `{review_sha}`.
Source receipt paths and SHA-256 values for conversion, board, float comparison, performance, and OE data
are recorded per row in `evidence-index.json`.

Nothing in this directory is read by the active release catalog or active `release/inputs.json`.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")
    refresh_note = {
        "schema_version": 1,
        "kind": "offline-yolo26-sseries-candidate-refresh",
        # The staging directory is temporary while the snapshot is built. Store
        # the immutable final directory name so metadata remains stable after rename.
        "snapshot_id": snapshot_id,
        "snapshot_status": "candidate-pending",
        "authoritative_audit": {"path": str(audit_path), "sha256": audit_sha,
                                "generated_at": audit["generated_at"], "ready": audit["ready"]},
        "review_matrix": {"path": str(review_path), "sha256": review_sha},
        "rows": 60,
        "checks": {key: audit["checks_passed"][key] for key in REQUIRED_AUDIT_CHECKS},
        "evidence_counts": {
            "valid_float_board_comparison": 60,
            "runtime_performance_1_and_2_threads": 60,
            "cpp_end_to_end_1_and_2_streams": cpp_count,
            "human_accuracy_delta_review_pending": 60,
            "human_release_approval_pending": 60,
        },
        "publication": {
            "uploaded": False,
            "active_catalog_modified": False,
            "candidate_status_changed": False,
            "oss_urls_populated": False,
            "human_approvals_populated": False,
        },
        "banner_claim": "AI-generated schematic; not model inference output",
    }
    write_json(staging / "refresh-audit.json", refresh_note)
    return cpp_count


def main() -> None:
    global STAGING, WORKBENCH, EXPERIMENT, CANDIDATE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True,
                        help="new path under candidate-staging, absolute or relative to candidate-staging")
    parser.add_argument("--audit-path", required=True, help="immutable authoritative audit JSON path")
    parser.add_argument("--review-path", required=True, help="immutable v4 manual review matrix JSON path")
    args = parser.parse_args()
    raw_audit_path = Path(args.audit_path).expanduser()
    raw_review_path = Path(args.review_path).expanduser()
    require(raw_audit_path.is_absolute() and not raw_audit_path.is_symlink(),
            "authoritative audit path must be absolute and may not be a symlink")
    require(raw_review_path.is_absolute() and not raw_review_path.is_symlink(),
            "review matrix path must be absolute and may not be a symlink")
    audit_path = raw_audit_path.resolve()
    review_path = raw_review_path.resolve()
    EXPERIMENT = audit_path.parent
    WORKBENCH = EXPERIMENT.parent.parent
    require(EXPERIMENT.parent.name == "experiments", "audit must be inside a Workbench experiments subdirectory")
    require(EXPERIMENT.is_dir(), f"workbench experiment is missing: {EXPERIMENT}")
    CANDIDATE_ROOT = WEB_ROOT / "candidate-staging"
    require_regular(audit_path, "authoritative S-series release audit")
    require_regular(review_path, "review matrix")
    audit_sha = sha256(audit_path)
    review_sha = sha256(review_path)
    audit = read_json(audit_path)
    snapshot_path, snapshot_id = resolve_snapshot_output(CANDIDATE_ROOT, args.snapshot_dir)
    final_path = snapshot_path
    require(not final_path.exists() and not final_path.is_symlink(),
            f"refusing to overwrite candidate snapshot: {final_path}")
    CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-tmp-", dir=CANDIDATE_ROOT))
    STAGING = tmp_path
    try:
        cpp_count = build_snapshot(tmp_path, snapshot_id, audit, audit_sha,
                                   audit_path, review_path, review_sha)
        require(sha256(audit_path) == audit_sha, "authoritative audit changed during snapshot generation")
        require(sha256(review_path) == review_sha, "review matrix changed during snapshot generation")
        require(not final_path.exists() and not final_path.is_symlink(),
                f"refusing to overwrite candidate snapshot: {final_path}")
        os.rename(tmp_path, final_path)
    except Exception:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    finally:
        STAGING = None
    print(f"staged immutable candidate snapshot: {final_path}")
    print(f"rows=60 comparator=60/60 runtime_perf=60/60 cpp_e2e={cpp_count}/60 human_approval=pending")


if __name__ == "__main__":
    main()
