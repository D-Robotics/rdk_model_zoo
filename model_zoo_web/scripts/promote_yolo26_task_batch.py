#!/usr/bin/env python3
"""Validate and optionally promote the 60 YOLO26 S-platform task releases.

The default mode is a read-only dry run. Promotion is all-or-nothing at the
batch level: all 60 workbench release gates, bundle hashes, OSS upload receipts,
task metrics and performance receipts must pass before any active file changes.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any, Callable
from urllib.parse import quote

import yaml


WEB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WEB_ROOT.parent
DEFAULT_WORKBENCH = REPO_ROOT.parent / "rdk_model_zoo_workbench"
TASKS = ("cls", "seg", "pose", "obb")
SIZES = ("n", "s", "m", "l", "x")
PLATFORMS = ("s600", "s100p", "s100")
OSS_BASE = "https://rdk-model-zoo.oss-cn-beijing.aliyuncs.com"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REPAIRED_KEYS = {("pose", "s", "s100p"), ("obb", "x", "s600"),
                 ("obb", "x", "s100p"), ("obb", "x", "s100")}

METRICS = {
    "cls": {"top1": "top1", "top5": "top5"},
    "seg": {"bbox/AP": "box_ap", "segm/AP": "mask_ap"},
    "pose": {"keypoints/AP": "keypoints_ap"},
    "obb": {"mAP50": "map_50"},
}
DATASET_LABELS = {
    "cls": "ImageNetV2 MatchedFrequency (not ILSVRC2012 val)",
    "seg": "COCO val2017 instance segmentation",
    "pose": "COCO val2017 person keypoints",
    "obb": "DOTA val",
}
TASK_DESCRIPTIONS = {
    "cls": {"zh": "图像分类", "en": "Image classification"},
    "seg": {"zh": "实例分割", "en": "Instance segmentation"},
    "pose": {"zh": "姿态估计", "en": "Pose estimation"},
    "obb": {"zh": "旋转框检测", "en": "Oriented object detection"},
}
TASK_LABELS = {
    "cls": "classification",
    "seg": "instance segmentation",
    "pose": "person keypoint estimation",
    "obb": "oriented object detection",
}
EXPECTED_OUTPUT_KINDS = {
    "cls": "topk_predictions",
    "seg": "instance_masks",
    "pose": "pose_instances",
    "obb": "rotated_boxes",
}


class PromotionError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise PromotionError(f"non-finite JSON value is forbidden: {value}")


def read_json(path: Path) -> dict[str, Any]:
    safe_regular(path, "JSON evidence")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"JSON root must be an object: {path}")
    return value


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionError(message)


def safe_regular(path: Path, label: str) -> None:
    need(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         f"{label}: missing, empty, or symlinked file: {path}")


def _expected_key(task: str, size: str, platform: str, name: str,
                  build_id: str | None = None) -> str:
    need(PurePosixPath(name).name == name and name not in (".", ".."),
         f"unsafe release bundle filename: {name!r}")
    prefix = f"models/ultralytics_yolo/yolo26/{task}/{size}/{platform}"
    if build_id is not None:
        need(BUILD_ID_RE.fullmatch(build_id) is not None, "unsafe rebuild BuildID")
        prefix += f"/rebuilds/{build_id}"
    return f"{prefix}/{name}"


def _upload_receipt_path(workbench: Path, key: str) -> Path:
    return (workbench / "outputs/receipts/uploads" /
            f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json")


def _verify_upload_receipt(workbench: Path, key: str, local_file: Path) -> str:
    receipt_path = _upload_receipt_path(workbench, key)
    safe_regular(receipt_path, "OSS upload receipt")
    receipt = read_json(receipt_path)
    expected_url = f"{OSS_BASE}/{quote(key, safe='/')}"
    need(receipt.get("status") == "verified", f"OSS receipt is not verified: {receipt_path}")
    need(receipt.get("key") == key, f"OSS receipt key mismatch: {receipt_path}")
    need(receipt.get("sha256") == sha256(local_file), f"OSS receipt SHA mismatch for {key}")
    need(receipt.get("size_bytes") == local_file.stat().st_size,
         f"OSS receipt size mismatch for {key}")
    need(receipt.get("acl") == "public-read", f"OSS release object must be public-read: {key}")
    need(receipt.get("url") == expected_url, f"OSS receipt URL mismatch for {key}")
    return expected_url


def _verify_bundle_upload_receipts(workbench: Path, task: str, size: str,
                                   platform: str, bundle: Path,
                                   artifact_name: str, build_id: str | None = None) -> dict[str, str]:
    urls: dict[str, str] = {}
    for object_name in (artifact_name, "oe_report.html", "oe_report_data.json",
                        "release.json", "SHA256SUMS"):
        key = _expected_key(task, size, platform, object_name, build_id)
        urls[object_name] = _verify_upload_receipt(workbench, key, bundle / object_name)
    return urls


def _verify_sums(bundle: Path, artifact_name: str) -> dict[str, str]:
    sums_path = bundle / "SHA256SUMS"
    safe_regular(sums_path, "SHA256SUMS")
    values: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        pieces = line.split("  ", 1)
        need(len(pieces) == 2 and SHA256_RE.fullmatch(pieces[0]) is not None,
             f"malformed SHA256SUMS line {line_number}: {sums_path}")
        digest, name = pieces
        need(PurePosixPath(name).name == name and name not in values,
             f"unsafe or duplicate SHA256SUMS filename {name!r}: {sums_path}")
        values[name] = digest
    expected_names = {artifact_name, "oe_report.html", "oe_report_data.json", "release.json"}
    need(set(values) == expected_names,
         f"SHA256SUMS file list mismatch in {bundle}: expected {sorted(expected_names)}, got {sorted(values)}")
    for name, expected in values.items():
        path = bundle / name
        safe_regular(path, name)
        need(sha256(path) == expected, f"SHA256SUMS mismatch: {path}")
    return values


def _verify_published_manifest(release: dict[str, Any], release_path: Path) -> None:
    """Require the re-packaged, externally published manifest for OSS-backed Web rows.

    The workbench stage manifest is intentionally not promotable: publication
    must finalize the manifest first, then regenerate SHA256SUMS and verified
    object receipts against those final bytes.
    """
    need(release.get("schema_version") == 2,
         f"unsupported release manifest schema: {release_path}")
    need(release.get("status") == "released" and
         release.get("publication_status") == "published",
         f"release manifest is not finalized as released/published: {release_path}")
    publication = release.get("publication")
    need(isinstance(publication, dict) and publication.get("schema_version") == 1 and
         publication.get("status") == "published" and
         "metadata_objects_pending_upload" not in publication,
         f"release publication metadata is invalid or still claims pending uploads: {release_path}")


def _finalized_bundle_path(workbench: Path, task: str, size: str, platform: str,
                            build_id: str | None = None) -> Path:
    path = (workbench / "releases/finalized/models/ultralytics_yolo/yolo26" /
            task / size / platform)
    if build_id is not None:
        need(BUILD_ID_RE.fullmatch(build_id) is not None, "unsafe rebuild BuildID")
        path = path / "rebuilds" / build_id
    return path


def _verify_finalization_receipt(workbench: Path, task: str, size: str, platform: str,
                                  bundle: Path, release: dict[str, Any],
                                  verified_urls: dict[str, str],
                                  build_id: str | None = None) -> None:
    """Cross-bind the v2 manifest, finalized files, source stage and all OSS receipts."""
    receipt_path = (workbench / "outputs/receipts/finalizations" /
                    f"{task}-{size}-{platform}{('-' + build_id) if build_id else ''}.json")
    safe_regular(receipt_path, "finalization receipt")
    receipt = read_json(receipt_path)
    need(receipt.get("schema_version") == 1 and
         receipt.get("kind") == "yolo26-task-release-finalization" and
         receipt.get("status") == "prepared-for-metadata-upload",
         f"unexpected finalization receipt schema/status: {receipt_path}")

    expected_model = {
        "source": "ultralytics_yolo", "provider": "ultralytics", "family": "yolo26",
        "task": task, "size": size,
    }
    need(release.get("model") == expected_model and receipt.get("model") == expected_model,
         f"finalization receipt model identity mismatch: {receipt_path}")
    target = release.get("target")
    need(isinstance(target, dict) and target.get("platform") == platform and
         receipt.get("target") == target,
         f"finalization receipt target identity mismatch: {receipt_path}")

    stage = receipt.get("stage")
    stage_bundle = _source_stage_path(workbench, task, size, platform, stage, build_id)
    stage_manifest_path = stage_bundle / "release.json"
    stage_sums_path = stage_bundle / "SHA256SUMS"
    safe_regular(stage_manifest_path, "source staged release manifest")
    safe_regular(stage_sums_path, "source staged SHA256SUMS")
    need(isinstance(stage, dict) and stage.get("path") == str(stage_bundle) and
         stage.get("release_manifest_sha256") == sha256(stage_manifest_path) and
         stage.get("sha256sums_sha256") == sha256(stage_sums_path),
         f"finalization receipt does not bind the current staged source: {receipt_path}")
    provenance = release.get("provenance", {})
    need(provenance.get("staged_release_manifest_sha256") == stage["release_manifest_sha256"] and
         provenance.get("staged_sha256sums_sha256") == stage["sha256sums_sha256"],
         f"released manifest staged-source hashes mismatch: {receipt_path}")

    artifact = release.get("artifact")
    need(isinstance(artifact, dict) and isinstance(artifact.get("name"), str),
         f"released artifact identity is missing: {receipt_path}")
    names = {
        "artifact": artifact["name"],
        "oe_report_html": "oe_report.html",
        "oe_report_data": "oe_report_data.json",
    }
    object_prefix = _expected_key(task, size, platform, "release.json", build_id).rsplit("/", 1)[0]
    publication = release["publication"]
    need(publication.get("oss_bucket") == "rdk-model-zoo" and
         publication.get("object_prefix") == object_prefix,
         f"release publication prefix/bucket mismatch: {receipt_path}")
    uploaded_objects = publication.get("uploaded_objects")
    receipt_uploaded_objects = receipt.get("uploaded_objects")
    receipt_refs = receipt.get("verified_upload_receipts")
    need(isinstance(uploaded_objects, dict) and set(uploaded_objects) == set(names) and
         isinstance(receipt_uploaded_objects, dict) and
         isinstance(receipt_refs, dict) and set(receipt_refs) == set(names),
         f"finalization receipt must bind exactly the three data-object uploads: {receipt_path}")

    expected_data_objects: dict[str, dict[str, Any]] = {}
    for role, name in names.items():
        key = _expected_key(task, size, platform, name, build_id)
        local_path = bundle / name
        upload_receipt = _upload_receipt_path(workbench, key)
        safe_regular(upload_receipt, f"{role} OSS upload receipt")
        expected_data_objects[role] = {
            "key": key,
            "sha256": sha256(local_path),
            "size_bytes": local_path.stat().st_size,
            "acl": "public-read",
            "url": verified_urls[name],
            "upload_receipt_sha256": sha256(upload_receipt),
        }
        need(uploaded_objects.get(role) == expected_data_objects[role] and
             receipt_uploaded_objects.get(role) == expected_data_objects[role],
             f"finalized manifest/receipt does not bind verified {role} upload: {receipt_path}")
        need(receipt_refs.get(role) == {
            "path": str(upload_receipt), "sha256": sha256(upload_receipt),
        }, f"finalization receipt upload-receipt path/hash mismatch for {role}: {receipt_path}")

    expected_metadata = [
        {"key": _expected_key(task, size, platform, "release.json", build_id),
         "name": "release.json", "acl": "public-read"},
        {"key": _expected_key(task, size, platform, "SHA256SUMS", build_id),
         "name": "SHA256SUMS", "acl": "public-read"},
    ]
    need(publication.get("metadata_objects") == expected_metadata,
         f"published manifest metadata-object declarations mismatch: {receipt_path}")
    need(receipt.get("metadata_objects_pending_upload") is True and
         receipt.get("pending_metadata_objects") == expected_metadata,
         f"local finalization receipt metadata-upload plan mismatch: {receipt_path}")

    artifact_urls = {
        "artifact": verified_urls[names["artifact"]],
        "release_manifest": verified_urls["release.json"],
        "checksums": verified_urls["SHA256SUMS"],
    }
    need(artifact.get("url") == artifact_urls["artifact"] and
         artifact.get("release_manifest_url") == artifact_urls["release_manifest"] and
         artifact.get("checksums_url") == artifact_urls["checksums"],
         f"released artifact URLs differ from verified OSS receipts: {receipt_path}")
    reports = release.get("reports", {})
    need(reports.get("oe_report_html_url") == verified_urls["oe_report.html"] and
         reports.get("oe_report_data_url") == verified_urls["oe_report_data.json"],
         f"released report URLs differ from verified OSS receipts: {receipt_path}")

    file_names = {artifact["name"], "oe_report.html", "oe_report_data.json",
                  "release.json", "SHA256SUMS"}
    finalized = receipt.get("finalized_bundle")
    need(isinstance(finalized, dict) and finalized.get("path") == str(bundle) and
         finalized.get("release_manifest_sha256") == sha256(bundle / "release.json") and
         finalized.get("sha256sums_sha256") == sha256(bundle / "SHA256SUMS"),
         f"finalization receipt path/manifest/checksum hash mismatch: {receipt_path}")
    file_records = finalized.get("files")
    need(isinstance(file_records, dict) and set(file_records) == file_names,
         f"finalization receipt must cover exactly five finalized files: {receipt_path}")
    for name in file_names:
        path = bundle / name
        record = file_records[name]
        need(isinstance(record, dict) and record == {
            "sha256": sha256(path), "size_bytes": path.stat().st_size,
        }, f"finalization receipt file hash/size mismatch for {name}: {receipt_path}")
    need(receipt.get("release_approval") == release.get("release_approval"),
         f"finalization receipt release approval mismatch: {receipt_path}")


def _source_stage_path(workbench: Path, task: str, size: str, platform: str,
                       stage: Any, build_id: str | None = None) -> Path:
    canonical = (workbench / "releases/models/ultralytics_yolo/yolo26" /
                 task / size / platform)
    expected = canonical if build_id is None else canonical / "rebuilds" / build_id
    allowed = {str(expected)}
    # The first Pose rebuild was staged before versioned staging existed. Its
    # finalization receipt and the selected gate still bind the exact bytes.
    if build_id is not None and (task, size, platform) == ("pose", "s", "s100p"):
        allowed.add(str(canonical))
    need(isinstance(stage, dict) and stage.get("path") in allowed,
         "finalization receipt has an unexpected source staged path")
    return Path(stage["path"])


def _metric_pair(task: str, accuracy: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    mapping = METRICS[task]
    board = accuracy.get("board")
    comparison = accuracy.get("comparison")
    need(isinstance(board, dict), f"{task} release accuracy.board is missing")
    need(isinstance(comparison, dict), f"{task} release accuracy.comparison is missing")
    need(set(board) == set(mapping), f"{task} board metric keys mismatch: {sorted(board)}")
    need(set(comparison) == set(mapping), f"{task} comparison metric keys mismatch: {sorted(comparison)}")
    float_metrics: dict[str, float] = {}
    runtime_metrics: dict[str, float] = {}
    for receipt_key, web_key in mapping.items():
        row = comparison[receipt_key]
        need(isinstance(row, dict), f"comparison metric {receipt_key} must be an object")
        need(set(row) >= {"float", "board", "board_minus_float", "retention_ratio"},
             f"comparison metric {receipt_key} lacks comparator values")
        float_value, board_value = row["float"], row["board"]
        for label, value in (("float", float_value), ("board", board_value),
                             ("board_minus_float", row["board_minus_float"])):
            need(isinstance(value, (int, float)) and not isinstance(value, bool) and
                 math.isfinite(float(value)), f"{task} {receipt_key}.{label} must be finite")
        need(0 <= float_value <= 1 and 0 <= board_value <= 1,
             f"{task} {receipt_key} accuracy must be a ratio between 0 and 1")
        need(math.isclose(board_value, board[receipt_key], rel_tol=0, abs_tol=1e-12),
             f"{task} board metric and comparator board value differ for {receipt_key}")
        need(math.isclose(row["board_minus_float"], board_value - float_value,
                          rel_tol=0, abs_tol=1e-12),
             f"{task} comparator delta is inconsistent for {receipt_key}")
        need(isinstance(row["retention_ratio"], (int, float)) and
             not isinstance(row["retention_ratio"], bool) and
             math.isfinite(float(row["retention_ratio"])),
             f"{task} comparator retention ratio is missing for {receipt_key}")
        float_metrics[web_key] = float(float_value)
        runtime_metrics[web_key] = float(board_value)
    return float_metrics, runtime_metrics


def _verify_accuracy(task: str, release: dict[str, Any], campaign: dict[str, Any]) -> dict[str, Any]:
    accuracy = release.get("accuracy")
    need(isinstance(accuracy, dict), f"{task}: release accuracy is missing")
    dataset = campaign.get("dataset", {})
    need(accuracy.get("dataset") == dataset.get("dataset"),
         f"{task}: release dataset does not match frozen campaign")
    expected_images = dataset.get("images")
    need(accuracy.get("expected_images") == expected_images and
         isinstance(expected_images, int) and expected_images > 0,
         f"{task}: release image count does not match frozen campaign")
    need(accuracy.get("comparison_status") == "valid",
         f"{task}: direct Float-vs-Runtime comparison is not valid")
    review = accuracy.get("human_review") or accuracy.get("review_record")
    need(isinstance(review, dict) and review.get("decision") == "accepted",
          f"{task}: accepted accuracy review is missing")
    reviewer_kind = review.get("reviewer_kind", "human")
    need(reviewer_kind in {"human", "agent"}, f"{task}: unknown accuracy reviewer kind")
    if reviewer_kind == "agent":
        approval = release.get("release_approval")
        need(isinstance(approval, dict) and approval.get("authorized") is True and
             approval.get("kind") == "explicit-user-web-publication-authorization",
             f"{task}: agent accuracy review requires explicit user publication authorization")
    need(isinstance(review.get("reviewed_by"), str) and review["reviewed_by"].strip(),
          f"{task}: accuracy reviewer is missing")
    need(isinstance(review.get("notes"), str) and len(review["notes"].strip()) >= 20,
          f"{task}: accuracy review notes are missing or too short")
    need(review.get("automatic_threshold_applied") is False,
         f"{task}: gate must not claim an automatic accuracy threshold")
    float_values, runtime_values = _metric_pair(task, accuracy)
    need(review.get("reviewed_metrics") == accuracy.get("comparison"),
          f"{task}: accuracy review does not bind the exact comparison metrics")
    model_accuracy = {
        "dataset": DATASET_LABELS[task],
        "task": TASK_LABELS[task],
        "images": expected_images,
        "float_onnx": float_values,
        "runtime": runtime_values,
    }
    if task == "obb":
        model_accuracy["evaluation_scope"] = "local_dota_val_single_scale"
        model_accuracy["scope_note"] = (
            "Local DOTA-v1.0 val evaluation at single scale over the validation split; "
            "not an official DOTA test-set score."
        )
    return model_accuracy


def _find_float_result(bench: Path, release: dict[str, Any],
                       workbench: Path | None = None) -> dict[str, Any]:
    provenance = release.get("provenance", {})
    expected_sha = provenance.get("float_result_sha256")
    matches = []
    path = bench / "output/float_result.json"
    if path.is_file() and not path.is_symlink() and sha256(path) == expected_sha:
        matches.append(path)
    reference_path = bench / "output/float_result_reference.json"
    reference_sha = provenance.get("float_result_reference_sha256")
    if reference_sha is not None:
        safe_regular(reference_path, "float result reference")
        need(sha256(reference_path) == reference_sha,
             f"float reference hash differs from release provenance: {reference_path}")
        reference = read_json(reference_path)
        source = reference.get("float_result", {})
        root = (workbench or bench.parents[2]).resolve()
        source_path = Path(str(source.get("path", "")))
        need(source_path.is_absolute() and source_path.resolve().is_relative_to(root) and
             reference.get("schema_version") == 1 and
             reference.get("kind") == "s-family-float-result-reference" and
             source.get("sha256") == expected_sha,
             f"float reference does not bind a safe historical baseline: {reference_path}")
        safe_regular(source_path, "historical float baseline")
        need(sha256(source_path) == expected_sha,
             f"historical float baseline changed: {source_path}")
        if source_path not in matches:
            matches.append(source_path)
    need(len(matches) == 1, f"float result hash does not uniquely bind to benchmark evidence: {bench}")
    return read_json(matches[0])


def _read_performance(workbench: Path, task: str, size: str, platform: str,
                      release: dict[str, Any], artifact_name: str,
                      benchmark: Path | None = None) -> dict[str, Any]:
    bench = benchmark or (workbench / "runs/benchmark" / f"yolo26-{task}-{size}-{platform}-releaseperf")
    perf_path = bench / "output/performance.json"
    input_path = bench / "input/perf-input.json"
    board_hash_path = bench / "logs/board-model.sha256"
    safe_regular(perf_path, "runtime performance")
    safe_regular(input_path, "fixed NV12 input manifest")
    safe_regular(board_hash_path, "board model hash")
    perf = read_json(perf_path)
    perf_input = read_json(input_path)
    environment_path = bench / "output/environment.json"
    safe_regular(environment_path, "performance environment")
    environment = read_json(environment_path)
    provenance = release.get("provenance", {})
    for path, field in ((perf_path, "runtime_performance_receipt_sha256"),
                        (input_path, "runtime_performance_input_sha256"),
                        (environment_path, "runtime_performance_environment_sha256")):
        need(sha256(path) == provenance.get(field),
             f"release does not bind current performance evidence ({field}): {path}")
    need(perf.get("tool") == "hrt_model_exec perf" and
         perf.get("implementation") == "native_cpp_cli" and
         perf.get("timing_scope") == "model_runtime" and
         perf.get("thread_semantics") == "runtime_submission_concurrency" and
         perf.get("core_id") == 1 and perf.get("input_kind") == "fixed_nv12" and
         perf.get("warmup_frames_per_condition", 0) >= 20 and
         perf.get("runs_per_condition") == 3 and perf.get("frames_per_run") == 200 and
         perf.get("fixed_input") == perf_input,
         f"runtime performance metadata or fixed-input binding is invalid: {perf_path}")
    need(perf_input.get("model_size") == [
        release.get("input", {}).get("width"), release.get("input", {}).get("height")
    ], f"runtime performance geometry does not match release manifest: {perf_path}")
    need(board_hash_path.read_text(encoding="utf-8").split()[0] ==
         release.get("artifact", {}).get("sha256"),
         f"runtime performance board model hash differs from release artifact: {board_hash_path}")
    raw_measurements = perf.get("raw_measurements")
    measurements = perf.get("measurements")
    need(isinstance(raw_measurements, list) and len(raw_measurements) == 6 and
         isinstance(measurements, list) and len(measurements) == 2 and
         {row.get("threads") for row in measurements if isinstance(row, dict)} == {1, 2},
         f"runtime performance must include all six runs and both concurrency summaries: {perf_path}")
    for row in measurements:
        for field in ("average_latency_ms", "observed_min_latency_ms", "observed_max_latency_ms", "aggregate_fps"):
            value = row.get(field)
            need(isinstance(value, (int, float)) and not isinstance(value, bool) and
                 math.isfinite(float(value)) and value > 0,
                 f"invalid runtime measurement {field}: {perf_path}")
        need(row["observed_min_latency_ms"] <= row["average_latency_ms"] <= row["observed_max_latency_ms"],
             f"runtime latency summary range is invalid: {perf_path}")
    need(perf.get("stages", {}).get("runtime") == "measured",
         f"runtime performance stage is not measured: {perf_path}")

    float_result = _find_float_result(bench, release, workbench)
    runtime_contract = float_result.get("runtime_contract", {})
    float_transport = runtime_contract.get("input", {}).get("float_transport", "")
    need("RGB float32 NCHW /255" in float_transport,
         f"float result does not establish RGB float32 NCHW /255 input: {bench}")
    need(float_result.get("model", {}).get("task") == task and
         float_result.get("model", {}).get("size") == size,
         f"float result identity differs from release: {bench}")

    width, height = release["input"]["width"], release["input"]["height"]
    performance = {
        "tool": perf["tool"],
        "implementation": perf["implementation"],
        "timing_scope": perf["timing_scope"],
        "thread_semantics": perf["thread_semantics"],
        "core_id": perf["core_id"],
        "warmup_frames_per_condition": perf["warmup_frames_per_condition"],
        "warmup_scope": perf.get("warmup_scope", "separate-process device warmup"),
        "runs_per_condition": perf["runs_per_condition"],
        "frames_per_run": perf["frames_per_run"],
        "input_kind": perf["input_kind"],
        "stages": perf["stages"],
        "measurements": copy.deepcopy(measurements),
        "end_to_end": [],
    }
    performance["fixed_input"] = {
        "source_image_sha256": perf_input.get("source_image_sha256"),
        "preprocessing": perf_input.get("preprocessing"),
        "resize_type": perf_input.get("resize_type"),
        "model_size": [width, height],
        "padding": perf_input.get("padding"),
        "planes": [
            {key: row[key] for key in ("name", "shape", "size_bytes", "sha256")}
            for row in perf_input.get("planes", [])
        ],
    }

    for streams in (1, 2):
        path = bench / "output" / f"e2e-{streams}.json"
        safe_regular(path, f"C++ end-to-end {streams}-stream result")
        row = read_json(path)
        need(sha256(path) == provenance.get(f"cpp_e2e_{streams}_receipt_sha256"),
             f"release does not bind the current C++ end-to-end receipt: {path}")
        need(row.get("schema_version") == 1 and
             row.get("pipeline_streams") == streams and
             row.get("runtime_submission_threads") == streams and
             row.get("output_kind") == EXPECTED_OUTPUT_KINDS[task] and
             row.get("model", "").split("/")[-1] == artifact_name and
             row.get("runtime_source_sha256") == provenance.get("cpp_runtime_source_sha256") and
             row.get("executable_sha256") == provenance.get("cpp_executable_sha256"),
             f"C++ end-to-end identity/provenance mismatch: {path}")
        end_to_end = {
            "tool": f"ultralytics_yolo_{task}",
            "implementation": row["implementation"],
            "timing_scope": row["timing_scope"],
            "pipeline_streams": row["pipeline_streams"],
            "runtime_submission_threads": row["runtime_submission_threads"],
            "cpu_thread_policy": row["cpu_thread_policy"],
            "online_cpu_threads": row["online_cpu_threads"],
            "opencv_threads": row["opencv_threads"],
            "cpu_governor": environment["cpu_governor"],
            "cpu_frequency_mhz": environment["cpu_frequency_mhz"],
            "bpu_frequency_mhz": environment["bpu_frequency_mhz"],
            "warmup_frames_per_round": row["warmup_frames_per_round"],
            "runs_per_round": row["runs_per_round"],
            "frames_per_round": row["frames_per_stream_per_round"],
            "rounds": row["rounds"],
            "timed_frames": row["timed_frames"],
            "aggregate_wall_ms": row["aggregate_wall_ms"],
            "metrics_ms": copy.deepcopy(row["metrics_ms"]),
            "throughput_fps": row["throughput_fps"],
        }
        need(row.get("frames_per_stream_per_round") == row.get("runs_per_round"),
             f"C++ end-to-end frame count does not match its receipt: {path}")
        performance["end_to_end"].append(end_to_end)
    return performance


def _verify_release_bundle(workbench: Path, task: str, size: str, platform: str,
                            campaign: dict[str, Any], campaign_path: Path,
                            override: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    build_id = override["build_id"] if override else None
    bundle = _finalized_bundle_path(workbench, task, size, platform, build_id)
    release_path = bundle / "release.json"
    safe_regular(release_path, "release manifest")
    release = read_json(release_path)
    expected_target = campaign.get("platforms", {}).get(platform, {})
    width, height = campaign.get("geometry", {}).get("size", [None, None])
    _verify_published_manifest(release, release_path)
    need(release.get("model") == {
        "source": "ultralytics_yolo", "provider": "ultralytics", "family": "yolo26",
        "task": task, "size": size,
    }, f"release model identity mismatch: {release_path}")
    need(release.get("target", {}).get("platform") == platform and
         release.get("target", {}).get("march") == expected_target.get("march") and
         release.get("target", {}).get("format") == expected_target.get("format"),
         f"release target mismatch with frozen campaign: {release_path}")
    need(release.get("input") == {"width": width, "height": height, "runtime_input": "nv12"},
         f"release geometry/runtime input mismatch: {release_path}")
    artifact = release.get("artifact")
    need(isinstance(artifact, dict), f"release artifact is missing: {release_path}")
    artifact_name = artifact.get("name")
    model_path = bundle / str(artifact_name)
    safe_regular(model_path, "compiled model")
    need(artifact.get("format") == expected_target.get("format") and
         artifact.get("size_bytes") == model_path.stat().st_size and
         artifact.get("sha256") == sha256(model_path),
         f"release artifact facts mismatch: {release_path}")
    report_map = release.get("reports", {})
    need(report_map.get("oe_report_html") == "oe_report.html" and
         report_map.get("oe_report_data") == "oe_report_data.json",
         f"release report filenames are unexpected: {release_path}")
    for name, sha_field in (("oe_report.html", "oe_report_html_sha256"),
                            ("oe_report_data.json", "oe_report_data_sha256")):
        path = bundle / name
        safe_regular(path, name)
        need(report_map.get(sha_field) == sha256(path), f"release report hash mismatch: {path}")
    oe_data = read_json(bundle / "oe_report_data.json")
    need(oe_data.get("provenance", {}).get("artifact_sha256") == artifact.get("sha256"),
         f"OE report is not bound to the released artifact: {bundle / 'oe_report_data.json'}")

    provenance = release.get("provenance")
    need(isinstance(provenance, dict), f"release provenance is missing: {release_path}")
    need(provenance.get("campaign_sha256") == sha256(campaign_path) and
         provenance.get("build_id") == campaign.get("build_id") and
         provenance.get("conversion_source_commit") == campaign.get("source_commit"),
         f"release provenance does not bind the frozen campaign/model: {release_path}")
    for field in ("runtime_source_manifest_sha256", "conversion_receipt_sha256",
                  "board_receipt_sha256", "float_result_sha256", "comparison_receipt_sha256",
                  "onnx_sha256", "oe_report_html_sha256", "oe_report_data_sha256",
                  "cpp_runtime_source_sha256", "cpp_executable_sha256"):
        need(SHA256_RE.fullmatch(str(provenance.get(field, ""))) is not None,
             f"release provenance SHA is missing or malformed ({field}): {release_path}")
    need(COMMIT_RE.fullmatch(str(provenance.get("runtime_source_commit", ""))) is not None,
         f"release runtime source commit is missing: {release_path}")
    approval = release.get("release_approval")
    if isinstance(approval, dict):
        _verify_agent_publication_authorization(workbench, task, size, platform, release, override)
    else:
        need(isinstance(approval, str) and approval.strip(), f"release approval is missing: {release_path}")

    bench = (override["benchmark"] if override else workbench / "runs/benchmark" /
             f"yolo26-{task}-{size}-{platform}-releaseperf")
    board_path = bench / "output/board_result.json.receipt.json"
    safe_regular(board_path, "board evaluation receipt")
    need(sha256(board_path) == provenance.get("board_receipt_sha256"),
         f"board evaluation receipt differs from staged release: {board_path}")
    board = read_json(board_path)
    conversion = board.get("conversion", {})
    build_path = Path(str(conversion.get("receipt_path", "")))
    need(build_path.is_absolute() and build_path.resolve().is_relative_to(workbench.resolve()),
         f"board receipt conversion path is unsafe: {board_path}")
    safe_regular(build_path, "conversion build receipt")
    need(sha256(build_path) == conversion.get("receipt_sha256") ==
         provenance.get("conversion_receipt_sha256"),
         f"conversion receipt hash differs from release and board evidence: {build_path}")
    build = read_json(build_path)
    need(board.get("model", {}).get("sha256") == artifact["sha256"] ==
         conversion.get("compiled_model_sha256") and
         build.get("artifacts", {}).get("model") == {
             "name": artifact_name, "sha256": artifact["sha256"],
             "size_bytes": artifact["size_bytes"]} and
         build.get("build_id") == provenance["build_id"] and
         build.get("target") == release["target"],
         f"release artifact does not bind the selected conversion and board model: {build_path}")
    comparison_path = bench / "output/comparison_receipt.json"
    safe_regular(comparison_path, "Float-vs-Runtime comparison receipt")
    need(sha256(comparison_path) == provenance.get("comparison_receipt_sha256"),
         f"comparison receipt differs from staged release: {comparison_path}")

    _verify_sums(bundle, str(artifact_name))
    urls = _verify_bundle_upload_receipts(workbench, task, size, platform, bundle, str(artifact_name), build_id)
    _verify_finalization_receipt(workbench, task, size, platform, bundle, release, urls, build_id)
    accuracy = _verify_accuracy(task, release, campaign)
    performance = _read_performance(workbench, task, size, platform, release, str(artifact_name), bench)
    return release, {"urls": urls, "oe_data": oe_data}, {"accuracy": accuracy, "performance": performance}


def _campaign(workbench: Path, task: str, size: str) -> tuple[dict[str, Any], Path]:
    path = workbench / "configs" / f"campaign-yolo26-{task}-{size}-v2.json"
    config = read_json(path)
    need(config.get("schema_version") == 4 and config.get("family") == "yolo26" and
         config.get("source") == "ultralytics_yolo" and config.get("task") == task and
         config.get("size") == size,
         f"frozen campaign identity mismatch: {path}")
    need(SHA256_RE.fullmatch(str(config.get("source_sha256", ""))) is not None and
         COMMIT_RE.fullmatch(str(config.get("source_commit", ""))) is not None,
         f"campaign source identity is malformed: {path}")
    return config, path


def _bound_reference(reference: dict[str, Any], workbench: Path, label: str) -> Path:
    need(isinstance(reference, dict) and isinstance(reference.get("path"), str) and
         SHA256_RE.fullmatch(str(reference.get("sha256", ""))) is not None,
         f"{label}: malformed bound file reference")
    path = Path(reference["path"])
    need(path.is_absolute() and path.resolve().is_relative_to(workbench.resolve()),
         f"{label}: evidence must stay inside the workbench")
    safe_regular(path, label)
    need(sha256(path) == reference["sha256"], f"{label}: SHA mismatch: {path}")
    return path


def _load_rebuild_overrides(path: Path | None, workbench: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if path is None:
        return {}
    manifest = read_json(path)
    records = manifest.get("records")
    need(manifest.get("schema_version") == 1 and isinstance(records, list),
         "rebuild selection manifest must contain schema 1 records")
    overrides = {}
    for record in records:
        need(isinstance(record, dict), "rebuild record must be an object")
        parts = str(record.get("release_id", "")).split("/")
        need(len(parts) == 5 and parts[:2] == ["ultralytics_yolo", "yolo26"],
             "rebuild record has invalid release identity")
        key = tuple(parts[2:])
        need(key in REPAIRED_KEYS and key not in overrides, "unknown or duplicate repaired release")
        task, size, platform = key
        build_id = record.get("build_id")
        need(isinstance(build_id, str) and BUILD_ID_RE.fullmatch(build_id) is not None,
             "rebuild record has unsafe BuildID")
        selection_path = _bound_reference((record.get("source_evidence_index") or {}).get("rebuild_selection"),
                                           workbench, "rebuild source selection")
        selected = read_json(selection_path)
        need(selected.get("schema_version") == 1 and
             selected.get("kind") == "yolo26-task-rebuild-selection" and
             selected.get("model") == {"family": "yolo26", "task": task, "size": size, "platform": platform},
             "rebuild selection identity mismatch")
        campaign_ref = selected.get("rebuild", {}).get("campaign")
        campaign_path = _bound_reference(campaign_ref, workbench, "rebuild campaign")
        campaign = read_json(campaign_path)
        need(campaign.get("build_id") == build_id and campaign_ref.get("build_id") == build_id,
             "rebuild BuildID differs from its frozen campaign")
        original, original_path = _campaign(workbench, task, size)
        baseline = selected.get("baseline", {}).get("campaign", {})
        need(baseline.get("path") == str(original_path) and baseline.get("sha256") == sha256(original_path) and
             campaign.get("source_sha256") == original.get("source_sha256") and
             campaign.get("geometry") == original.get("geometry"),
             "rebuild must bind the original checkpoint and input geometry")
        benchmark = Path(selected.get("rebuild", {}).get("benchmark_path", ""))
        need(benchmark.is_absolute() and benchmark.resolve().is_relative_to(workbench.resolve()) and
             benchmark.is_dir(), "rebuild benchmark is missing or outside the workbench")
        files = record.get("files") or {}
        uploads = record.get("upload_receipts") or {}
        need(set(files) == {"artifact_file", "oe_report_html", "oe_report_data", "release_manifest", "checksums"} and
             set(uploads) == {"artifact", "oe_report_html", "oe_report_data", "release_manifest", "checksums"},
             "rebuild record must bind all five package files and upload receipts")
        bundle = _finalized_bundle_path(workbench, task, size, platform, build_id)
        for role, reference in files.items():
            file = _bound_reference(reference, workbench, f"rebuild package {role}")
            need(file.parent == bundle, "rebuild package file is outside its selected bundle")
        need(Path(files["release_manifest"]["path"]).name == "release.json" and
             files["artifact_file"]["sha256"] == selected["rebuild"]["conversion_receipt"]["model_sha256"],
             "rebuild package does not bind the selected model")
        for role, reference in uploads.items():
            _bound_reference(reference, workbench, f"rebuild upload receipt {role}")
        overrides[key] = {"build_id": build_id, "selection_path": selection_path,
                          "selection": selected, "campaign_path": campaign_path,
                          "campaign": campaign, "benchmark": benchmark}
    need(set(overrides) == REPAIRED_KEYS, "rebuild selection must cover exactly the four repaired releases")
    return overrides


def _verify_agent_publication_authorization(workbench: Path, task: str, size: str, platform: str,
                                            release: dict[str, Any], override: dict[str, Any] | None) -> None:
    authorization = release["release_approval"]
    provenance = release["provenance"]
    binding = {"task": task, "size": size, "platform": platform,
               "campaign_sha256": provenance.get("campaign_sha256"),
               "runtime_source_manifest_sha256": provenance.get("runtime_source_manifest_sha256"),
               "model_sha256": release["artifact"]["sha256"],
               "board_receipt_sha256": provenance.get("board_receipt_sha256"),
               "float_result_sha256": provenance.get("float_result_sha256"),
               "comparison_receipt_sha256": provenance.get("comparison_receipt_sha256")}
    if override:
        binding["rebuild_selection_sha256"] = sha256(override["selection_path"])
        binding["baseline_campaign_sha256"] = override["selection"]["baseline"]["campaign"]["sha256"]
        binding["board_runtime_source_manifest_sha256"] = override["selection"]["rebuild"]["runtime_source_manifest"]["sha256"]
    need(authorization.get("kind") == "explicit-user-web-publication-authorization" and
         authorization.get("source") == "conversation:user-message" and
         authorization.get("authorized") is True and authorization.get("scope") == binding and
         authorization.get("recorded_by") == "Codex agent",
         "agent publication authorization does not bind this exact release")
    source_path = _bound_reference({"path": authorization.get("source_record_path"),
                                   "sha256": authorization.get("source_record_sha256")},
                                   workbench, "user publication authorization record")
    source = read_json(source_path)
    need(source.get("schema_version") == 1 and source.get("source") == "explicit_user_message_in_current_conversation" and
         source.get("request_text") == authorization.get("instruction") and
         "发布" in str(source.get("request_text", "")) and
         "update_web" in source.get("authorized_actions", []) and
         any((r.get("task"), r.get("size"), r.get("platform")) == (task, size, platform)
             for r in source.get("scope", []) if isinstance(r, dict)),
         "source authorization record does not cover this Web release")


def _make_platform_record(task: str, size: str, platform: str, campaign: dict[str, Any],
                          release: dict[str, Any], verified: dict[str, Any],
                          derived: dict[str, Any], release_date: str) -> dict[str, Any]:
    artifact = release["artifact"]
    accuracy = derived["accuracy"]
    performance = derived["performance"]
    platform_details = campaign["platforms"][platform]
    return {
        "platform": platform,
        "status": "released",
        "released_at": release_date,
        "artifact": {
            "format": artifact["format"],
            "march": release["target"]["march"],
            "runtime_input": release["input"]["runtime_input"],
            "url": verified["urls"][artifact["name"]],
            "release_manifest_url": verified["urls"]["release.json"],
            "checksums_url": verified["urls"]["SHA256SUMS"],
            "size_bytes": artifact["size_bytes"],
            "sha256": artifact["sha256"],
        },
        "reports": {"oe_conversion_url": verified["urls"]["oe_report.html"]},
        "accuracy": accuracy,
        "performance": performance,
        "provenance": {
            "repository_commit": release["provenance"]["conversion_source_commit"],
            "build_id": release["provenance"]["build_id"],
            "runtime_source_commit": release["provenance"]["runtime_source_commit"],
            "runtime_source_manifest_sha256": release["provenance"]["runtime_source_manifest_sha256"],
            "conversion_receipt_sha256": release["provenance"]["conversion_receipt_sha256"],
            "board_receipt_sha256": release["provenance"]["board_receipt_sha256"],
            "float_result_sha256": release["provenance"]["float_result_sha256"],
            "comparison_receipt_sha256": release["provenance"]["comparison_receipt_sha256"],
            "campaign_source_sha256": campaign["source_sha256"],
            "source_checkpoint_sha256": campaign["source_sha256"],
            "source_weight_url": campaign["source_url"],
            "precision_policy": campaign["precision_policy"],
            "platform_image_id": platform_details["image_id"],
        },
    }


def _build_record(task: str, workbench: Path, release_date: str,
                   gate_auditor: Callable[[str, str, str], tuple[bool, str]],
                   rebuild_overrides: dict | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes], list[str]]:
    record = {
        "schema_version": 1,
        "id": f"ultralytics_yolo/yolo26/{task}",
        "name": f"YOLO26 {task.upper()}",
        "domain": "vision",
        "source": "ultralytics_yolo",
        "provider": "ultralytics",
        "family": "yolo26",
        "task": task,
        "description": TASK_DESCRIPTIONS[task],
        "license": {},
        "sample_path": "samples/vision/ultralytics_yolo",
        "variants": [],
    }
    input_releases: dict[str, Any] = {}
    report_files: dict[str, bytes] = {}
    errors: list[str] = []
    for size in SIZES:
        try:
            campaign, campaign_path = _campaign(workbench, task, size)
            model = campaign["model"]
            license_info = campaign["license"]
            record["license"] = {"name": license_info["name"], "url": license_info["url"]}
            width, height = campaign["geometry"]["size"]
            source_input = {
                "format": "rgb", "dtype": "float32", "layout": "NCHW",
                "shape": [1, 3, height, width], "scale": 1 / 255,
            }
            variant = {
                "size": size,
                "model": {
                    "parameter_count": model["parameter_count"],
                    "gflops": model["gflops"],
                    "gflops_method": model["measurement_method"],
                },
                "input": {"width": width, "height": height, "source": source_input},
                "platforms": [],
            }
            for platform in PLATFORMS:
                ready, gate_reason = gate_auditor(task, size, platform)
                if not ready:
                    errors.append(f"{task}/{size}/{platform}: task_release_gate failed: {gate_reason}")
                    continue
                override = (rebuild_overrides or {}).get((task, size, platform))
                selected_campaign = override["campaign"] if override else campaign
                selected_campaign_path = override["campaign_path"] if override else campaign_path
                release, verified, derived = _verify_release_bundle(
                    workbench, task, size, platform, selected_campaign, selected_campaign_path, override)
                variant["platforms"].append(_make_platform_record(
                    task, size, platform, selected_campaign, release, verified, derived, release_date))
                report_name = f"yolo26-{task}-{size}-{platform}-oe-data.json"
                report_files[report_name] = (
                    json.dumps(verified["oe_data"], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                ).encode("utf-8")
                input_releases[f"{record['id']}/{size}/{platform}"] = {
                    "oe_data": f"reports/{report_name}",
                }
        except (PromotionError, KeyError, TypeError, IndexError, OSError) as exc:
            errors.append(f"{task}/{size}: {exc}")
            continue
        record["variants"].append(variant)
    need(bool(record["license"].get("name")) and bool(record["license"].get("url")),
         f"{task}: missing campaign license information")
    return record, input_releases, report_files, errors


def _stage_gate_auditor(workbench: Path, rebuild_overrides: dict | None = None) -> Callable[[str, str, str], tuple[bool, str]]:
    gate_path = workbench / "scripts/stage_task_release.py"
    safe_regular(gate_path, "stage_task_release.py")
    scripts_path = str(workbench / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    spec = importlib.util.spec_from_file_location("promotion_stage_task_release_gate", gate_path)
    need(spec is not None and spec.loader is not None, f"cannot load release gate: {gate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = workbench.resolve()

    def audit(task: str, size: str, platform: str) -> tuple[bool, str]:
        try:
            override = (rebuild_overrides or {}).get((task, size, platform))
            result = module.Audit(task, size, platform,
                                  selection_path=override["selection_path"] if override else None).audit()
            failed = [row.get("name") for row in result.get("checks", [])
                      if row.get("passed") is not True]
            return bool(result.get("ready")), "all gates pass" if not failed else ", ".join(failed)
        except Exception as exc:  # report the gate failure; never bypass it
            return False, f"gate audit error: {exc}"
    return audit


def _load_catalog_builder():
    if str(WEB_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(WEB_ROOT / "scripts"))
    import build_catalog
    return build_catalog


def make_plan(workbench: Path, release_date: str,
               gate_auditor: Callable[[str, str, str], tuple[bool, str]] | None = None,
               rebuild_selection: Path | None = None) -> tuple[dict[Path, bytes], list[str], dict[str, Any]]:
    workbench = workbench.expanduser().resolve()
    need(workbench.is_dir(), f"workbench directory does not exist: {workbench}")
    rebuild_overrides = _load_rebuild_overrides(rebuild_selection, workbench)
    if gate_auditor is None:
        gate_auditor = _stage_gate_auditor(workbench, rebuild_overrides)

    release_inputs_path = WEB_ROOT / "release/inputs.json"
    release_inputs = read_json(release_inputs_path)
    need(release_inputs.get("schema_version") == 1 and
         isinstance(release_inputs.get("models"), dict) and
         isinstance(release_inputs.get("releases"), dict),
         f"active release inputs schema is invalid: {release_inputs_path}")
    detect_releases_before = {
        key: value for key, value in release_inputs["releases"].items()
        if key.startswith("ultralytics_yolo/yolo26/detect/") or
           key.startswith("ultralytics_yolo/yolo11/detect/")
    }
    need(len(detect_releases_before) == 40,
         f"expected the existing 40 Detect release mappings; found {len(detect_releases_before)}")

    banner_manifest_path = WEB_ROOT / "release/assets/yolo26-real-task-banners.json"
    banner_manifest = read_json(banner_manifest_path)
    for task in TASKS:
        item = banner_manifest.get("items", {}).get(task)
        need(isinstance(item, dict), f"task banner manifest lacks {task}")
        banner = WEB_ROOT / "release/assets" / item.get("file", "")
        safe_regular(banner, f"{task} task banner")
        need(banner.stat().st_size == item.get("size_bytes") and sha256(banner) == item.get("sha256"),
             f"task banner integrity check failed: {banner}")

    models = copy.deepcopy(release_inputs["models"])
    releases = copy.deepcopy(release_inputs["releases"])
    writes: dict[Path, bytes] = {}
    errors: list[str] = []
    gate_checks = 0
    catalog_builder = _load_catalog_builder()
    for task in TASKS:
        record, task_releases, reports, task_errors = _build_record(
            task, workbench, release_date, gate_auditor, rebuild_overrides)
        errors.extend(task_errors)
        gate_checks += len(SIZES) * len(PLATFORMS)
        try:
            target_path = WEB_ROOT / "data/vision/ultralytics_yolo/yolo26" / f"{task}.yaml"
            catalog_builder.validate_record(record, target_path)
        except Exception as exc:
            errors.append(f"{task}: active catalog schema validation failed: {exc}")
        models[record["id"]] = {
            "cover": "assets/" + banner_manifest["items"][task]["file"],
            "cover_label": banner_manifest["items"][task]["cover_label"],
        }
        releases.update(task_releases)
        for filename, data in reports.items():
            target = WEB_ROOT / "release/reports" / filename
            if target.exists() or target.is_symlink():
                errors.append(f"promotion report target already exists: {target}")
            writes[target] = data
        record_path = WEB_ROOT / "data/vision/ultralytics_yolo/yolo26" / f"{task}.yaml"
        if record_path.exists() or record_path.is_symlink():
            errors.append(f"promotion YAML target already exists: {record_path}")
        writes[record_path] = yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=100).encode("utf-8")

    new_inputs = {"schema_version": 1, "models": models, "releases": releases}
    need(all(new_inputs["releases"].get(key) == value
             for key, value in detect_releases_before.items()),
         "existing Detect input mappings changed while composing promotion plan")
    input_bytes = (json.dumps(new_inputs, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    writes[release_inputs_path] = input_bytes
    summary = {
        "entry_count": gate_checks,
        "task_count": len(TASKS),
        "release_receipts_expected": len(TASKS) * len(SIZES) * len(PLATFORMS) * 5,
        "files_to_write": len(writes),
        "detect_release_mappings_preserved": len(detect_releases_before),
        "release_inputs_path": str(release_inputs_path),
        "repaired_release_builds": {
            "/".join(key): value["build_id"] for key, value in rebuild_overrides.items()
        },
    }
    return writes, errors, summary


def _commit_plan(writes: dict[Path, bytes]) -> None:
    release_inputs_path = WEB_ROOT / "release/inputs.json"
    targets = list(writes)
    need(targets and targets[-1] == release_inputs_path,
         "release/inputs.json must remain the final activation file")
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != release_inputs_path:
            need(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    original = {release_inputs_path: release_inputs_path.read_bytes()}
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target, payload in writes.items():
            fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.promotion-", dir=target.parent)
            temporary = Path(temporary_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged[target] = temporary
        for target in targets:
            if target != release_inputs_path:
                need(not target.exists() and not target.is_symlink(),
                     f"promotion target appeared during staging; aborting: {target}")
            if target == release_inputs_path:
                need(target.read_bytes() == original[target],
                     "active release/inputs.json changed during promotion; aborting")
            os.replace(staged[target], target)
            committed.append(target)
    except Exception:
        for target in reversed(committed):
            if target in original:
                fd, backup_name = tempfile.mkstemp(prefix=f".{target.name}.rollback-", dir=target.parent)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(original[target])
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(backup_name, target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench-root", type=Path, default=DEFAULT_WORKBENCH,
                        help="read-only workbench evidence root")
    parser.add_argument("--release-date", help="catalog release date YYYY-MM-DD; required with --apply")
    parser.add_argument("--rebuild-selection", type=Path,
                        help="hash-bound manifest selecting the four already published repairs")
    parser.add_argument("--apply", action="store_true",
                        help="write all 60 releases only after every gate and receipt passes")
    return parser


def main(argv: list[str] | None = None) -> int:
    if sys.flags.optimize:
        raise RuntimeError("Do not run release promotion checks with Python -O")
    args = build_parser().parse_args(argv)
    if args.release_date:
        try:
            release_date = dt.date.fromisoformat(args.release_date).isoformat()
        except ValueError as exc:
            raise SystemExit("--release-date must use YYYY-MM-DD") from exc
    else:
        release_date = dt.date.today().isoformat()
    if args.apply and not args.release_date:
        raise SystemExit("--apply requires an explicit --release-date")
    writes, errors, summary = make_plan(args.workbench_root, release_date,
                                      rebuild_selection=args.rebuild_selection)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run",
                      "summary": summary,
                      "planned_paths": [str(path.relative_to(WEB_ROOT)) for path in writes],
                      "blockers": errors}, ensure_ascii=False, indent=2))
    if errors:
        print(f"promotion blocked: {len(errors)} gate/evidence/schema issue(s)", file=sys.stderr)
        return 2
    if args.apply:
        _commit_plan(writes)
        print("promoted all 60 YOLO26 task releases")
    else:
        print("dry-run passed; no files were changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
