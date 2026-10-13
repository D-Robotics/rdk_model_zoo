#!/usr/bin/env python3
"""Build the samples-only Model Zoo Web catalog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import yaml


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "model_zoo_web"
DATA_ROOT = WEB_ROOT / "data"
BUILD_ROOT = WEB_ROOT / "build"
CATALOG_PATH = BUILD_ROOT / "catalog.json"
META_PATH = BUILD_ROOT / "catalog.meta.json"
OSS_HOST = "rdk-model-zoo.oss-cn-beijing.aliyuncs.com"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Samples live on develop; the Web branch does not have to carry their source. When a sample
# directory is absent from this checkout, it is verified against this git ref instead.
SAMPLE_REF_ENV = "MODEL_ZOO_SAMPLE_REF"
SAMPLE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SIZE_ORDER = {name: index for index, name in enumerate(("n", "s", "m", "l", "x"))}
# Sample-profile records describe deployable repository samples (for example CPU
# trackers or detector samples without a formal model release yet). They render
# on the same dashboard but never pass as released neural models: their platform
# status is "available" and the released-model fields stay forbidden below.
SAMPLE_PROFILE = "sample"
SAMPLE_STATUS = "available"
SAMPLE_PLATFORMS = ("x5", "s100", "s100p", "s600")
SAMPLE_PIPELINE_KINDS = ("cpu_tracker", "detector_cpu_tracker", "neural_detector", "avatar_pipeline")
SAMPLE_ARTIFACT_HOSTS = ("archive.d-robotics.cc", "codeload.github.com")
# Release manifests at the sample commit are what the samples' download
# scripts fetch from; descriptor-bound records must publish the same URLs.
SAMPLE_RELEASE_MANIFESTS = {
    "x5": "docs/release/x5/models.yaml",
    "s100": "docs/release/s/models.yaml",
    "s100p": "docs/release/s/models.yaml",
    "s600": "docs/release/s/models.yaml",
}
SAMPLE_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# Fields that only a released neural-model record may carry. A sample profile
# that lists any of them is rejected instead of silently ignoring the data.
SAMPLE_FORBIDDEN_VARIANT_FIELDS = ("model", "input", "performance", "accuracy")
SAMPLE_FORBIDDEN_PLATFORM_FIELDS = (
    "artifact", "reports", "accuracy", "performance", "released_at",
)
ACCURACY_SCHEMA_PATH = WEB_ROOT / "accuracy-metrics.json"
ACCURACY_SCHEMA = json.loads(ACCURACY_SCHEMA_PATH.read_text(encoding="utf-8"))


class CatalogError(ValueError):
    pass


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{label} must be a mapping")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise CatalogError(f"{label} must be a non-empty list")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"{label} must be a non-empty string")
    return value


def validate_localized_text(value: Any, label: str) -> None:
    localized = require_mapping(value, label)
    require_string(localized.get("zh"), f"{label}.zh")
    require_string(localized.get("en"), f"{label}.en")


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise CatalogError(f"{label} must be finite")
    return number


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogError(f"{label} must be a positive integer")
    return value


def require_positive_number(value: Any, label: str) -> float:
    number = require_number(value, label)
    if number <= 0:
        raise CatalogError(f"{label} must be greater than zero")
    return number


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogError(f"{label} must be a non-negative integer")
    return value


def validate_sample_path(value: Any, label: str) -> str:
    sample_path = require_string(value, label)
    pure_path = PurePosixPath(sample_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.parts[0] != "samples":
        raise CatalogError(f"{label} must be a safe path below samples/")
    if (ROOT / sample_path).is_dir():
        return sample_path
    ref = os.environ.get(SAMPLE_REF_ENV, "")
    if not ref:
        raise CatalogError(
            f"{label} does not exist: {sample_path} "
            f"(set {SAMPLE_REF_ENV} to a git ref such as origin/develop to verify it there)"
        )
    if not SAMPLE_REF_RE.match(ref):
        raise CatalogError(f"{SAMPLE_REF_ENV} is not a plain git ref: {ref!r}")
    kind = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-t", f"{ref}:{sample_path}"],
        capture_output=True, text=True, check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "tree":
        raise CatalogError(f"{label} does not exist locally or in {ref}: {sample_path}")
    return sample_path


def verify_commit_tree(commit: str, sample_path: str, label: str) -> None:
    """Bind sample_commit:sample_path to a real git tree, not just a shape.

    The pinned commit must exist in this repository (CI fetches develop) and
    the sample path must resolve inside it, otherwise the detail-page source
    link 404s while the catalog still validates.
    """
    kind = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-t", f"{commit}:{sample_path}"],
        capture_output=True, text=True, check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "tree":
        raise CatalogError(
            f"{label}: {commit}:{sample_path} is not a tree in this repository; "
            "sample_commit must be a real commit that contains sample_path"
        )


def validate_artifact(
    artifact: dict[str, Any],
    *,
    source: str,
    family: str,
    task: str,
    size: str,
    platform: str,
    label: str,
    build_id: str | None = None,
) -> None:
    url = require_string(artifact.get("url"), f"{label}.url")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != OSS_HOST or parsed.query or parsed.fragment:
        raise CatalogError(f"{label}.url must be a direct HTTPS URL on {OSS_HOST}")
    filename = PurePosixPath(unquote(parsed.path)).name
    expected_path = f"/models/{source}/{family}/{task}/{size}/{platform}/{filename}"
    allowed_paths = {expected_path}
    if isinstance(build_id, str) and BUILD_ID_RE.fullmatch(build_id):
        allowed_paths.add(
            f"/models/{source}/{family}/{task}/{size}/{platform}/rebuilds/{build_id}/{filename}"
        )
    if unquote(parsed.path) not in allowed_paths:
        raise CatalogError(f"{label}.url must use {expected_path}")

    sha256 = require_string(artifact.get("sha256"), f"{label}.sha256")
    if not SHA256_RE.fullmatch(sha256):
        raise CatalogError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
    size_bytes = artifact.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise CatalogError(f"{label}.size_bytes must be a positive integer")

    base_url = url.rsplit("/", 1)[0]
    for field, expected_name in (
        ("release_manifest_url", "release.json"),
        ("checksums_url", "SHA256SUMS"),
    ):
        expected_url = f"{base_url}/{expected_name}"
        if artifact.get(field) != expected_url:
            raise CatalogError(f"{label}.{field} must be {expected_url}")


def validate_end_to_end_record(end_to_end: dict[str, Any], label: str) -> int:
    require_string(end_to_end.get("tool"), f"{label}.tool")
    require_string(end_to_end.get("implementation"), f"{label}.implementation")
    require_string(end_to_end.get("timing_scope"), f"{label}.timing_scope")
    pipeline_streams = require_positive_int(
        end_to_end.get("pipeline_streams"),
        f"{label}.pipeline_streams",
    )
    submission_threads = require_positive_int(
        end_to_end.get("runtime_submission_threads"),
        f"{label}.runtime_submission_threads",
    )
    if submission_threads != pipeline_streams:
        raise CatalogError(
            f"{label}.runtime_submission_threads must equal pipeline_streams"
        )
    cpu_thread_policy = require_string(
        end_to_end.get("cpu_thread_policy"),
        f"{label}.cpu_thread_policy",
    )
    if cpu_thread_policy not in {"all_online", "fixed"}:
        raise CatalogError(f"{label}.cpu_thread_policy must be all_online or fixed")
    online_cpu_threads = require_positive_int(
        end_to_end.get("online_cpu_threads"),
        f"{label}.online_cpu_threads",
    )
    opencv_threads = require_positive_int(
        end_to_end.get("opencv_threads"),
        f"{label}.opencv_threads",
    )
    if cpu_thread_policy == "all_online" and opencv_threads != online_cpu_threads:
        raise CatalogError(
            f"{label}.opencv_threads must equal online_cpu_threads "
            "for the all_online policy"
        )
    require_string(end_to_end.get("cpu_governor"), f"{label}.cpu_governor")
    require_positive_number(
        end_to_end.get("cpu_frequency_mhz"),
        f"{label}.cpu_frequency_mhz",
    )
    require_positive_number(
        end_to_end.get("bpu_frequency_mhz"),
        f"{label}.bpu_frequency_mhz",
    )
    require_nonnegative_int(
        end_to_end.get("warmup_frames_per_round"),
        f"{label}.warmup_frames_per_round",
    )
    rounds = require_positive_int(end_to_end.get("rounds"), f"{label}.rounds")
    frames_per_round = require_positive_int(
        end_to_end.get("frames_per_round"),
        f"{label}.frames_per_round",
    )

    metrics_label = f"{label}.metrics_ms"
    metrics = require_mapping(end_to_end.get("metrics_ms"), metrics_label)
    metric_means: dict[str, float] = {}
    for stage in ("preprocess", "runtime", "postprocess", "end_to_end"):
        metric_label = f"{metrics_label}.{stage}"
        metric = require_mapping(metrics.get(stage), metric_label)
        values = {
            field: require_number(metric.get(field), f"{metric_label}.{field}")
            for field in ("mean", "p50", "p95", "min", "max")
        }
        if min(values.values()) <= 0:
            raise CatalogError(f"{metric_label} values must be greater than zero")
        if not values["min"] <= values["p50"] <= values["p95"] <= values["max"]:
            raise CatalogError(f"{metric_label} percentiles must be ordered")
        if not values["min"] <= values["mean"] <= values["max"]:
            raise CatalogError(f"{metric_label}.mean must be within the observed range")
        metric_means[stage] = values["mean"]

    stage_sum = sum(metric_means[stage] for stage in ("preprocess", "runtime", "postprocess"))
    if not math.isclose(stage_sum, metric_means["end_to_end"], abs_tol=0.001):
        raise CatalogError(f"{metrics_label} stage means must sum to end_to_end.mean")
    throughput_fps = require_number(
        end_to_end.get("throughput_fps"),
        f"{label}.throughput_fps",
    )
    if throughput_fps <= 0:
        raise CatalogError(f"{label}.throughput_fps must be greater than zero")

    has_wall_evidence = (
        end_to_end.get("timed_frames") is not None
        or end_to_end.get("aggregate_wall_ms") is not None
    )
    if has_wall_evidence or pipeline_streams > 1:
        timed_frames = require_positive_int(
            end_to_end.get("timed_frames"),
            f"{label}.timed_frames",
        )
        expected_frames = pipeline_streams * frames_per_round * rounds
        if timed_frames != expected_frames:
            raise CatalogError(f"{label}.timed_frames must be {expected_frames}")
        aggregate_wall_ms = require_number(
            end_to_end.get("aggregate_wall_ms"),
            f"{label}.aggregate_wall_ms",
        )
        if aggregate_wall_ms <= 0:
            raise CatalogError(f"{label}.aggregate_wall_ms must be greater than zero")
        expected_fps = timed_frames * 1000.0 / aggregate_wall_ms
    else:
        expected_fps = 1000.0 / metric_means["end_to_end"]
    if not math.isclose(throughput_fps, expected_fps, rel_tol=0.001):
        raise CatalogError(f"{label}.throughput_fps does not match its timing evidence")
    return pipeline_streams


def validate_accuracy(value: Any, task: str, label: str) -> None:
    accuracy = require_mapping(value, label)
    require_string(accuracy.get("dataset"), f"{label}.dataset")
    require_string(accuracy.get("task"), f"{label}.task")
    require_positive_int(accuracy.get("images"), f"{label}.images")

    canonical_task = ACCURACY_SCHEMA.get("task_aliases", {}).get(task, task)
    profile = ACCURACY_SCHEMA.get("tasks", {}).get(canonical_task)
    if not isinstance(profile, dict):
        supported = ", ".join(sorted(ACCURACY_SCHEMA.get("tasks", {})))
        raise CatalogError(f"{label}: unsupported model task {task!r}; expected one of {supported}")

    expected_dataset = profile.get("dataset")
    if expected_dataset and accuracy["dataset"] != expected_dataset:
        raise CatalogError(f"{label}.dataset must be {expected_dataset}")

    evaluation_scope = profile.get("evaluation_scope")
    if evaluation_scope:
        expected_scope = evaluation_scope["value"]
        actual_scope = require_string(accuracy.get("evaluation_scope"), f"{label}.evaluation_scope")
        if actual_scope != expected_scope:
            raise CatalogError(f"{label}.evaluation_scope must be {expected_scope}")

    for stage in ("float_onnx", "runtime"):
        stage_label = f"{label}.{stage}"
        stage_accuracy = require_mapping(accuracy.get(stage), stage_label)
        for field in profile["required"]:
            metric_label = f"{stage_label}.{field}"
            metric_value = require_number(stage_accuracy.get(field), metric_label)
            if not 0 <= metric_value <= 1:
                raise CatalogError(f"{metric_label} must be a ratio between 0 and 1")


def validate_localized_note(value: Any, label: str, *, required: bool = True) -> None:
    if value is None:
        if required:
            raise CatalogError(f"{label} is required")
        return
    validate_localized_text(value, label)


def validate_sample_artifact(artifact: dict[str, Any], label: str) -> None:
    require_string(artifact.get("role"), f"{label}.role")
    require_string(artifact.get("filename"), f"{label}.filename")
    filename = PurePosixPath(artifact["filename"])
    if filename.is_absolute() or ".." in filename.parts:
        raise CatalogError(f"{label}.filename must be a safe relative file name")
    require_string(artifact.get("format"), f"{label}.format")
    url = require_string(artifact.get("url"), f"{label}.url")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in SAMPLE_ARTIFACT_HOSTS or parsed.query or parsed.fragment:
        raise CatalogError(
            f"{label}.url must be a direct stable HTTPS URL on one of "
            f"{', '.join(SAMPLE_ARTIFACT_HOSTS)}"
        )
    provenance = artifact.get("sha256_provenance")
    if provenance not in ("publisher", "observed-on-board", "not-provided"):
        raise CatalogError(
            f"{label}.sha256_provenance must be publisher, observed-on-board or not-provided"
        )
    sha256 = artifact.get("sha256")
    if provenance == "not-provided":
        if sha256 is not None:
            raise CatalogError(f"{label}.sha256 must be omitted when the provenance is not-provided")
    else:
        if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
            raise CatalogError(
                f"{label}.sha256 must be 64 lowercase hexadecimal characters "
                f"(provenance {provenance})"
            )
    size_bytes = artifact.get("size_bytes")
    if size_bytes is not None and (isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0):
        raise CatalogError(f"{label}.size_bytes must be a positive integer or omitted")


def validate_sample_link(link: dict[str, Any], label: str) -> None:
    validate_localized_note(link.get("label"), f"{label}.label")
    url = require_string(link.get("url"), f"{label}.url")
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme or parsed.netloc or url.startswith("/") or ".." in PurePosixPath(url).parts:
        raise CatalogError(f"{label}.url must be an absolute HTTPS URL or a site-relative path")
    parts = PurePosixPath(url).parts
    if len(parts) != 2 or parts[0] != "sample-evidence" or not parts[1].endswith((".json", ".log")):
        raise CatalogError(f"{label}.url must be an HTTPS URL or a sample-evidence/<file> asset path")
    # The link ships with the site; fail at catalog time when the file is absent.
    if not (WEB_ROOT / "release" / url).is_file():
        raise CatalogError(f"{label}.url: {url} does not exist in model_zoo_web/release")


def validate_sample_platform(platform_record: dict[str, Any], platform_label: str) -> None:
    platform = require_string(platform_record.get("platform"), f"{platform_label}.platform")
    if platform not in SAMPLE_PLATFORMS:
        raise CatalogError(
            f"{platform_label}.platform must be one of {', '.join(SAMPLE_PLATFORMS)} "
            "(x3 is archive-only)"
        )
    if platform_record.get("status") != SAMPLE_STATUS:
        raise CatalogError(
            f"{platform_label}.status must be {SAMPLE_STATUS}; a sample profile never marks "
            "a platform released. Promote the record to the released-model schema instead."
        )
    forbidden = [key for key in SAMPLE_FORBIDDEN_PLATFORM_FIELDS if key in platform_record]
    if forbidden:
        raise CatalogError(
            f"{platform_label} must not carry released-model fields: {', '.join(forbidden)}. "
            "Sample profiles and released models are separate records."
        )
    runtimes = platform_record.get("runtimes")
    if not isinstance(runtimes, list) or not runtimes or any(item not in ("python", "cpp") for item in runtimes):
        raise CatalogError(f"{platform_label}.runtimes must list python and/or cpp")
    availability = require_mapping(
        platform_record.get("model_availability"), f"{platform_label}.model_availability"
    )
    status = availability.get("status")
    if status not in ("published-download", "manual-compile", "no-neural-artifact"):
        raise CatalogError(
            f"{platform_label}.model_availability.status must be published-download, "
            "manual-compile or no-neural-artifact"
        )
    validate_localized_note(availability.get("note"), f"{platform_label}.model_availability.note")
    if status == "published-download":
        artifacts = platform_record.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise CatalogError(
                f"{platform_label}.artifacts must list at least one published artifact "
                "when model_availability.status is published-download"
            )
        for index, artifact_value in enumerate(artifacts):
            artifact = require_mapping(artifact_value, f"{platform_label}.artifacts[{index}]")
            validate_sample_artifact(artifact, f"{platform_label}.artifacts[{index}]")
        # A split detector publishes its PartA/PartB pair or nothing: one half
        # alone is unusable on the board. Neural artifacts follow the delivery
        # line's published format (bin on x5, hbm on the S series) so a file
        # cannot be registered against the wrong target.
        pair_roles = {artifact["role"] for artifact in artifacts if artifact["role"] in ("parta", "partb")}
        if pair_roles and pair_roles != {"parta", "partb"}:
            raise CatalogError(
                f"{platform_label}: a published split detector must carry both parta and partb"
            )
        expected_format = "bin" if platform == "x5" else "hbm"
        for artifact in artifacts:
            if artifact["role"] in ("parta", "partb", "detector") and artifact["format"] != expected_format:
                raise CatalogError(
                    f"{platform_label}.artifacts: {artifact['role']} must be a {expected_format} file "
                    f"on {platform}, not {artifact['format']}"
                )
    elif status == "no-neural-artifact":
        artifacts = platform_record.get("artifacts")
        if artifacts is not None:
            # A tracker without a neural model may still publish its pinned
            # upstream source, but nothing that looks like a deployable model.
            if not isinstance(artifacts, list) or not artifacts:
                raise CatalogError(
                    f"{platform_label}.artifacts must be a non-empty list when present"
                )
            for index, artifact_value in enumerate(artifacts):
                artifact = require_mapping(artifact_value, f"{platform_label}.artifacts[{index}]")
                if artifact.get("role") != "upstream-source":
                    raise CatalogError(
                        f"{platform_label}.artifacts[{index}].role must be upstream-source "
                        "for a platform without a neural artifact"
                    )
                if artifact.get("format") in ("bin", "hbm", "onnx", "pt"):
                    raise CatalogError(
                        f"{platform_label}.artifacts[{index}] must not look like a "
                        "deployable model artifact"
                    )
                validate_sample_artifact(artifact, f"{platform_label}.artifacts[{index}]")
    elif platform_record.get("artifacts") is not None:
        raise CatalogError(
            f"{platform_label}.artifacts must be omitted unless the platform offers a "
            "published download"
        )
    if status == "manual-compile":
        guide_url = availability.get("guide_url")
        if not isinstance(guide_url, str) or not guide_url.strip():
            raise CatalogError(
                f"{platform_label}.model_availability.guide_url is required for manual-compile"
            )
        parsed_guide = urlparse(guide_url)
        if parsed_guide.scheme != "https" or not parsed_guide.netloc:
            raise CatalogError(
                f"{platform_label}.model_availability.guide_url must be an absolute HTTPS URL"
            )
    validate_localized_note(platform_record.get("verification"), f"{platform_label}.verification")
    evidence = platform_record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise CatalogError(f"{platform_label}.evidence must be a non-empty list of links")
    for index, link_value in enumerate(evidence):
        link = require_mapping(link_value, f"{platform_label}.evidence[{index}]")
        validate_sample_link(link, f"{platform_label}.evidence[{index}]")
    platform_quickstart = platform_record.get("quickstart")
    if platform_quickstart is not None:
        # Platforms may override the record-level steps so the shown commands
        # match the selected hardware (e.g. manual model preparation on S100P).
        if not isinstance(platform_quickstart, list) or not platform_quickstart:
            raise CatalogError(f"{platform_label}.quickstart must be a non-empty list when present")
        for index, entry_value in enumerate(platform_quickstart):
            entry = require_mapping(entry_value, f"{platform_label}.quickstart[{index}]")
            validate_localized_note(entry.get("title"), f"{platform_label}.quickstart[{index}].title")
            require_string(entry.get("command"), f"{platform_label}.quickstart[{index}].command")


def validate_sample_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    relative_source = source_path.relative_to(ROOT).as_posix()
    if record.get("schema_version") != 1:
        raise CatalogError(f"{relative_source}: schema_version must be 1")

    record_id = require_string(record.get("id"), f"{relative_source}.id")
    domain = require_string(record.get("domain"), f"{relative_source}.domain")
    source = require_string(record.get("source"), f"{relative_source}.source")
    provider = require_string(record.get("provider"), f"{relative_source}.provider")
    family = require_string(record.get("family"), f"{relative_source}.family")
    task = require_string(record.get("task"), f"{relative_source}.task")
    require_string(record.get("name"), f"{relative_source}.name")
    validate_localized_text(record.get("description"), f"{relative_source}.description")
    license_info = require_mapping(record.get("license"), f"{relative_source}.license")
    require_string(license_info.get("name"), f"{relative_source}.license.name")
    license_url = require_string(license_info.get("url"), f"{relative_source}.license.url")
    parsed_license_url = urlparse(license_url)
    if parsed_license_url.scheme != "https" or not parsed_license_url.netloc:
        raise CatalogError(f"{relative_source}.license.url must be an absolute HTTPS URL")
    expected_id = f"{source}/{family}/{task}"
    if record_id != expected_id:
        raise CatalogError(f"{relative_source}.id must be {expected_id}")
    expected_source = DATA_ROOT / domain / source / family / f"{task}.yaml"
    if source_path != expected_source:
        raise CatalogError(f"{relative_source}: expected path {expected_source.relative_to(ROOT)}")
    validate_sample_path(record.get("sample_path"), f"{relative_source}.sample_path")
    sample_commit = require_string(record.get("sample_commit"), f"{relative_source}.sample_commit")
    if not SAMPLE_COMMIT_RE.fullmatch(sample_commit):
        raise CatalogError(
            f"{relative_source}.sample_commit must be the pinned 40-character develop commit"
        )
    verify_commit_tree(sample_commit, record["sample_path"], relative_source)
    descriptor_rel = record.get("artifact_descriptor")
    if descriptor_rel is not None:
        descriptor_rel = require_string(descriptor_rel, f"{relative_source}.artifact_descriptor")
        pure_descriptor = PurePosixPath(descriptor_rel)
        sample_parts = PurePosixPath(record["sample_path"]).parts
        if pure_descriptor.is_absolute() or ".." in pure_descriptor.parts \
                or pure_descriptor.parts[: len(sample_parts)] != sample_parts:
            raise CatalogError(
                f"{relative_source}.artifact_descriptor must be a path below sample_path"
            )
    if record.get("pipeline") not in SAMPLE_PIPELINE_KINDS:
        raise CatalogError(
            f"{relative_source}.pipeline must be one of {', '.join(SAMPLE_PIPELINE_KINDS)}"
        )
    validate_localized_note(record.get("pipeline_note"), f"{relative_source}.pipeline_note")
    task_ids = record.get("task_ids")
    if task_ids is not None:
        if not isinstance(task_ids, list) or not task_ids:
            raise CatalogError(f"{relative_source}.task_ids must be a non-empty list when present")
        for index, task_id in enumerate(task_ids):
            if not isinstance(task_id, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", task_id):
                raise CatalogError(
                    f"{relative_source}.task_ids[{index}] must be a kebab-case task id"
                )

    variants = require_list(record.get("variants"), f"{relative_source}.variants")
    seen_sizes: set[str] = set()
    published_by_platform: dict[str, dict[str, tuple[str, str]]] = {}
    for variant_index, variant_value in enumerate(variants):
        variant_label = f"{relative_source}.variants[{variant_index}]"
        variant = require_mapping(variant_value, variant_label)
        size = require_string(variant.get("size"), f"{variant_label}.size")
        if size in seen_sizes:
            raise CatalogError(f"{relative_source}: duplicate variant size {size}")
        seen_sizes.add(size)
        forbidden = [key for key in SAMPLE_FORBIDDEN_VARIANT_FIELDS if key in variant]
        if forbidden:
            raise CatalogError(
                f"{variant_label} must not carry released-model fields: {', '.join(forbidden)}"
            )
        spec_rows = variant.get("input_contract")
        if spec_rows is not None:
            if not isinstance(spec_rows, list) or not spec_rows:
                raise CatalogError(f"{variant_label}.input_contract must be a non-empty list")
            for row_index, row_value in enumerate(spec_rows):
                row = require_mapping(row_value, f"{variant_label}.input_contract[{row_index}]")
                # Contract rows render on both the zh and en pages, so both the
                # label and the value must be localized mappings.
                validate_localized_text(
                    row.get("label"), f"{variant_label}.input_contract[{row_index}].label"
                )
                validate_localized_text(
                    row.get("value"), f"{variant_label}.input_contract[{row_index}].value"
                )
        platforms = require_list(variant.get("platforms"), f"{variant_label}.platforms")
        seen_platforms: set[str] = set()
        for platform_index, platform_value in enumerate(platforms):
            platform_label = f"{variant_label}.platforms[{platform_index}]"
            platform_record = require_mapping(platform_value, platform_label)
            platform = require_string(platform_record.get("platform"), f"{platform_label}.platform")
            if platform in seen_platforms:
                raise CatalogError(f"{variant_label}: duplicate platform {platform}")
            seen_platforms.add(platform)
            validate_sample_platform(platform_record, platform_label)
            if record.get("quickstart") is None and platform_record.get("quickstart") is None:
                raise CatalogError(
                    f"{platform_label}: no quickstart; provide it on the platform or the record "
                    "so the shown commands match the selected hardware"
                )
            roles = published_by_platform.setdefault(platform, {})
            for artifact in platform_record.get("artifacts") or []:
                # Artifacts without a digest (sha256_provenance: not-provided)
                # cannot be hash-bound to the descriptor; skip them here.
                if artifact.get("role") in ("parta", "partb", "detector") and artifact.get("sha256"):
                    roles[artifact["role"]] = (artifact["filename"], artifact["sha256"], artifact["url"])

    if descriptor_rel is not None:
        # Bind every published neural artifact to the descriptor pinned in the
        # sample source, so a stale or mislabelled file cannot register.
        show = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{sample_commit}:{descriptor_rel}"],
            capture_output=True, text=True, check=False,
        )
        if show.returncode != 0:
            raise CatalogError(
                f"{relative_source}.artifact_descriptor: {sample_commit}:{descriptor_rel} "
                "cannot be read from this repository"
            )
        try:
            descriptor = json.loads(show.stdout)
        except json.JSONDecodeError:
            raise CatalogError(f"{relative_source}.artifact_descriptor must be a JSON file") from None
        targets = descriptor.get("targets") if isinstance(descriptor, dict) else None
        if not isinstance(targets, dict):
            raise CatalogError(
                f"{relative_source}.artifact_descriptor must carry targets.<platform>.<role> "
                "entries with file and sha256"
            )
        manifest_cache: dict[str, dict[str, dict[str, Any]]] = {}
        for platform, roles in sorted(published_by_platform.items()):
            target = targets.get(platform)
            if not isinstance(target, dict):
                raise CatalogError(
                    f"{relative_source}.artifact_descriptor has no targets.{platform} entry"
                )
            for role, (filename, sha256, _url) in sorted(roles.items()):
                entry = target.get(role)
                # The descriptor stores bare file names; a record may prefix the
                # platform subdirectory. The digest is the binding that matters.
                if not isinstance(entry, dict) or entry.get("file") != PurePosixPath(filename).name \
                        or entry.get("sha256") != sha256:
                    raise CatalogError(
                        f"{relative_source}: published {role} on {platform} does not match the "
                        f"pinned descriptor {descriptor_rel}"
                    )
            # The download URL must come from the delivery manifest the sample's
            # downloader fetches from, so a verified filename/digest cannot be
            # pointed at another file's URL (e.g. the S600 part under S100).
            manifest_rel = SAMPLE_RELEASE_MANIFESTS[platform]
            if manifest_rel not in manifest_cache:
                manifest_show = subprocess.run(
                    ["git", "-C", str(ROOT), "show", f"{sample_commit}:{manifest_rel}"],
                    capture_output=True, text=True, check=False,
                )
                if manifest_show.returncode != 0:
                    raise CatalogError(
                        f"{relative_source}: {sample_commit}:{manifest_rel} cannot be read "
                        "from this repository"
                    )
                manifest_models = yaml.safe_load(manifest_show.stdout) or {}
                by_sample_path = {
                    entry.get("sample_path"): entry
                    for entry in manifest_models.get("models") or []
                    if isinstance(entry, dict)
                }
                manifest_cache[manifest_rel] = by_sample_path
            manifest_entry = manifest_cache[manifest_rel].get(record["sample_path"])
            if manifest_entry is None:
                raise CatalogError(
                    f"{relative_source}: {manifest_rel} at {sample_commit} has no entry for "
                    f"{record['sample_path']}"
                )
            manifest_assets = {
                asset.get("filename"): asset
                for asset in manifest_entry.get("assets") or []
                if isinstance(asset, dict)
            }
            for role, (filename, sha256, url) in sorted(roles.items()):
                asset = manifest_assets.get(filename)
                if asset is None:
                    raise CatalogError(
                        f"{relative_source}: published {role} on {platform} ({filename}) is not "
                        f"listed in {manifest_rel} at {sample_commit}"
                    )
                if asset.get("url") != url or asset.get("sha256") != sha256:
                    raise CatalogError(
                        f"{relative_source}: published {role} on {platform} does not match the "
                        f"pinned release manifest {manifest_rel}"
                    )

    for field in ("quickstart", "resources"):
        entries = record.get(field)
        if entries is None:
            continue
        if not isinstance(entries, list) or not entries:
            raise CatalogError(f"{relative_source}.{field} must be a non-empty list when present")
        for index, entry_value in enumerate(entries):
            entry = require_mapping(entry_value, f"{relative_source}.{field}[{index}]")
            if field == "quickstart":
                validate_localized_note(entry.get("title"), f"{relative_source}.{field}[{index}].title")
                require_string(entry.get("command"), f"{relative_source}.{field}[{index}].command")
            else:
                validate_sample_link(entry, f"{relative_source}.{field}[{index}]")
    missing = require_list(record.get("missing"), f"{relative_source}.missing")
    allowed_status = ("missing", "not-applicable")
    for index, item_value in enumerate(missing):
        item = require_mapping(item_value, f"{relative_source}.missing[{index}]")
        require_string(item.get("field"), f"{relative_source}.missing[{index}].field")
        if item.get("status") not in allowed_status:
            raise CatalogError(
                f"{relative_source}.missing[{index}].status must be one of {', '.join(allowed_status)}"
            )
        # The field id stays machine-readable; the page shows this title.
        validate_localized_note(item.get("title"), f"{relative_source}.missing[{index}].title")
        validate_localized_note(item.get("note"), f"{relative_source}.missing[{index}].note")

    normalized = copy.deepcopy(record)
    normalized["source_file"] = relative_source
    return normalized


def validate_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    relative_source = source_path.relative_to(ROOT).as_posix()
    if record.get("schema_version") != 1:
        raise CatalogError(f"{relative_source}: schema_version must be 1")
    profile = record.get("profile")
    if profile is not None:
        # The released-model schema keeps no profile marker; the sample profile is
        # dispatched before this validator runs. Anything else is a typo, not a model.
        raise CatalogError(
            f"{relative_source}: unknown profile {profile!r}; "
            f"only {SAMPLE_PROFILE!r} is a separate profile"
        )

    record_id = require_string(record.get("id"), f"{relative_source}.id")
    domain = require_string(record.get("domain"), f"{relative_source}.domain")
    source = require_string(record.get("source"), f"{relative_source}.source")
    provider = require_string(record.get("provider"), f"{relative_source}.provider")
    family = require_string(record.get("family"), f"{relative_source}.family")
    task = require_string(record.get("task"), f"{relative_source}.task")
    validate_localized_text(record.get("description"), f"{relative_source}.description")
    license_info = require_mapping(record.get("license"), f"{relative_source}.license")
    require_string(license_info.get("name"), f"{relative_source}.license.name")
    license_url = require_string(license_info.get("url"), f"{relative_source}.license.url")
    parsed_license_url = urlparse(license_url)
    if parsed_license_url.scheme != "https" or not parsed_license_url.netloc:
        raise CatalogError(f"{relative_source}.license.url must be an absolute HTTPS URL")
    expected_id = f"{source}/{family}/{task}"
    if record_id != expected_id:
        raise CatalogError(f"{relative_source}.id must be {expected_id}")

    expected_source = DATA_ROOT / domain / source / family / f"{task}.yaml"
    if source_path != expected_source:
        raise CatalogError(f"{relative_source}: expected path {expected_source.relative_to(ROOT)}")
    validate_sample_path(record.get("sample_path"), f"{relative_source}.sample_path")

    variants = require_list(record.get("variants"), f"{relative_source}.variants")
    seen_sizes: set[str] = set()
    for variant_index, variant_value in enumerate(variants):
        variant_label = f"{relative_source}.variants[{variant_index}]"
        variant = require_mapping(variant_value, variant_label)
        size = require_string(variant.get("size"), f"{variant_label}.size")
        if size in seen_sizes:
            raise CatalogError(f"{relative_source}: duplicate variant size {size}")
        seen_sizes.add(size)

        input_config = require_mapping(variant.get("input"), f"{variant_label}.input")
        for dimension in ("width", "height"):
            require_positive_int(input_config.get(dimension), f"{variant_label}.input.{dimension}")

        model_info = require_mapping(variant.get("model"), f"{variant_label}.model")
        require_positive_int(model_info.get("parameter_count"), f"{variant_label}.model.parameter_count")
        if require_number(model_info.get("gflops"), f"{variant_label}.model.gflops") <= 0:
            raise CatalogError(f"{variant_label}.model.gflops must be greater than zero")

        source_input = require_mapping(input_config.get("source"), f"{variant_label}.input.source")
        require_string(source_input.get("format"), f"{variant_label}.input.source.format")
        require_string(source_input.get("dtype"), f"{variant_label}.input.source.dtype")
        require_string(source_input.get("layout"), f"{variant_label}.input.source.layout")
        source_shape = require_list(source_input.get("shape"), f"{variant_label}.input.source.shape")
        if len(source_shape) != 4:
            raise CatalogError(f"{variant_label}.input.source.shape must contain four dimensions")
        for dimension_index, dimension in enumerate(source_shape):
            require_positive_int(dimension, f"{variant_label}.input.source.shape[{dimension_index}]")
        if source_shape[-2:] != [input_config["height"], input_config["width"]]:
            raise CatalogError(f"{variant_label}.input.source.shape must match input width and height")
        if require_number(source_input.get("scale"), f"{variant_label}.input.source.scale") <= 0:
            raise CatalogError(f"{variant_label}.input.source.scale must be greater than zero")

        platforms = require_list(variant.get("platforms"), f"{variant_label}.platforms")
        seen_platforms: set[str] = set()
        for platform_index, platform_value in enumerate(platforms):
            platform_label = f"{variant_label}.platforms[{platform_index}]"
            platform_record = require_mapping(platform_value, platform_label)
            platform = require_string(platform_record.get("platform"), f"{platform_label}.platform")
            if platform in seen_platforms:
                raise CatalogError(f"{variant_label}: duplicate platform {platform}")
            seen_platforms.add(platform)
            if platform_record.get("status") != "released":
                raise CatalogError(f"{platform_label}.status must be released")
            require_string(platform_record.get("released_at"), f"{platform_label}.released_at")
            artifact = require_mapping(platform_record.get("artifact"), f"{platform_label}.artifact")
            validate_artifact(
                artifact,
                source=source,
                family=family,
                task=task,
                size=size,
                platform=platform,
                label=f"{platform_label}.artifact",
                build_id=(platform_record.get("provenance") or {}).get("build_id"),
            )

            reports = require_mapping(platform_record.get("reports"), f"{platform_label}.reports")
            release_base_url = artifact["url"].rsplit("/", 1)[0]
            expected_report_url = f"{release_base_url}/oe_report.html"
            if reports.get("oe_conversion_url") != expected_report_url:
                raise CatalogError(
                    f"{platform_label}.reports.oe_conversion_url must be {expected_report_url}"
                )

            validate_accuracy(
                platform_record.get("accuracy"),
                task,
                f"{platform_label}.accuracy",
            )
            performance = require_mapping(platform_record.get("performance"), f"{platform_label}.performance")
            require_string(performance.get("tool"), f"{platform_label}.performance.tool")
            require_string(performance.get("implementation"), f"{platform_label}.performance.implementation")
            require_string(performance.get("timing_scope"), f"{platform_label}.performance.timing_scope")
            thread_semantics = require_string(
                performance.get("thread_semantics"),
                f"{platform_label}.performance.thread_semantics",
            )
            if thread_semantics != "runtime_submission_concurrency":
                raise CatalogError(
                    f"{platform_label}.performance.thread_semantics must be "
                    "runtime_submission_concurrency"
                )
            require_nonnegative_int(performance.get("core_id"), f"{platform_label}.performance.core_id")
            require_nonnegative_int(
                performance.get("warmup_frames_per_condition"),
                f"{platform_label}.performance.warmup_frames_per_condition",
            )
            require_positive_int(
                performance.get("runs_per_condition"),
                f"{platform_label}.performance.runs_per_condition",
            )
            require_positive_int(performance.get("frames_per_run"), f"{platform_label}.performance.frames_per_run")

            stages = require_mapping(performance.get("stages"), f"{platform_label}.performance.stages")
            for stage in ("preprocess", "runtime", "postprocess", "end_to_end"):
                status = require_string(stages.get(stage), f"{platform_label}.performance.stages.{stage}")
                if status not in {"measured", "not_measured"}:
                    raise CatalogError(
                        f"{platform_label}.performance.stages.{stage} must be measured or not_measured"
                    )
            if stages["runtime"] != "measured":
                raise CatalogError(f"{platform_label}.performance.stages.runtime must be measured")

            measurements = require_list(
                performance.get("measurements"),
                f"{platform_label}.performance.measurements",
            )
            seen_threads: set[int] = set()
            for measurement_index, measurement_value in enumerate(measurements):
                measurement_label = f"{platform_label}.performance.measurements[{measurement_index}]"
                measurement = require_mapping(measurement_value, measurement_label)
                threads = require_positive_int(measurement.get("threads"), f"{measurement_label}.threads")
                if threads in seen_threads:
                    raise CatalogError(f"{platform_label}.performance: duplicate thread count {threads}")
                seen_threads.add(threads)
                latency = require_number(
                    measurement.get("average_latency_ms"),
                    f"{measurement_label}.average_latency_ms",
                )
                minimum = require_number(
                    measurement.get("observed_min_latency_ms"),
                    f"{measurement_label}.observed_min_latency_ms",
                )
                maximum = require_number(
                    measurement.get("observed_max_latency_ms"),
                    f"{measurement_label}.observed_max_latency_ms",
                )
                fps = require_number(measurement.get("aggregate_fps"), f"{measurement_label}.aggregate_fps")
                if min(latency, minimum, maximum, fps) <= 0:
                    raise CatalogError(f"{measurement_label} values must be greater than zero")
                if not minimum <= latency <= maximum:
                    raise CatalogError(f"{measurement_label} latency must be within the observed range")

            end_to_end_label = f"{platform_label}.performance.end_to_end"
            end_to_end_value = performance.get("end_to_end")
            if isinstance(end_to_end_value, dict):
                end_to_end_records = [end_to_end_value]
            else:
                end_to_end_records = require_list(end_to_end_value, end_to_end_label)
            seen_pipeline_streams: set[int] = set()
            for condition_index, condition_value in enumerate(end_to_end_records):
                condition_label = f"{end_to_end_label}[{condition_index}]"
                condition = require_mapping(condition_value, condition_label)
                pipeline_streams = validate_end_to_end_record(condition, condition_label)
                if pipeline_streams in seen_pipeline_streams:
                    raise CatalogError(
                        f"{end_to_end_label}: duplicate pipeline_streams {pipeline_streams}"
                    )
                seen_pipeline_streams.add(pipeline_streams)

    normalized = copy.deepcopy(record)
    normalized["source_file"] = relative_source
    # Sizes outside the n/s/m/l/x ladder (for example conv-small-224) order by parameter count.
    normalized["variants"].sort(
        key=lambda item: (SIZE_ORDER.get(item["size"], 99), item["model"]["parameter_count"], item["size"])
    )
    for variant in normalized["variants"]:
        variant["platforms"].sort(key=lambda item: item["platform"])
    return normalized


def build_catalog() -> tuple[bytes, bytes]:
    source_paths = sorted(
        path for path in DATA_ROOT.rglob("*.yaml")
        if not any(part.startswith(".") for part in path.relative_to(DATA_ROOT).parts)
    )
    if not source_paths:
        raise CatalogError(f"no YAML records found below {DATA_ROOT}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    release_count = 0
    sample_count = 0
    for source_path in source_paths:
        with source_path.open(encoding="utf-8") as handle:
            record = require_mapping(yaml.safe_load(handle), source_path.relative_to(ROOT).as_posix())
        if record.get("profile") == SAMPLE_PROFILE:
            normalized = validate_sample_record(record, source_path)
            sample_count += 1
        else:
            normalized = validate_record(record, source_path)
            release_count += sum(len(variant["platforms"]) for variant in normalized["variants"])
        if normalized["id"] in seen_ids:
            raise CatalogError(f"duplicate model id: {normalized['id']}")
        seen_ids.add(normalized["id"])
        records.append(normalized)

    records.sort(key=lambda item: item["id"])
    model_count = len(records) - sample_count
    catalog = {
        "schema_version": 1,
        "source": "model_zoo_web/data",
        "model_count": model_count,
        "release_count": release_count,
        "sample_count": sample_count,
        "models": records,
    }
    catalog_bytes = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode()
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    metadata = {
        "schema_version": 1,
        "catalog": "catalog.json",
        "sha256": catalog_sha256,
        "bytes": len(catalog_bytes),
        "model_count": model_count,
        "release_count": release_count,
        "sample_count": sample_count,
        "generator": "model_zoo_web/scripts/build_catalog.py",
    }
    metadata_bytes = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode()
    return catalog_bytes, metadata_bytes


def write_or_check(catalog_bytes: bytes, metadata_bytes: bytes, check: bool) -> None:
    expected = ((CATALOG_PATH, catalog_bytes), (META_PATH, metadata_bytes))
    if check:
        for path, content in expected:
            if not path.is_file() or path.read_bytes() != content:
                raise CatalogError(f"generated artifact is stale: {path.relative_to(ROOT)}")
        return

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    for path, content in expected:
        path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify existing output without rewriting it")
    args = parser.parse_args()
    try:
        catalog_bytes, metadata_bytes = build_catalog()
        write_or_check(catalog_bytes, metadata_bytes, args.check)
    except (CatalogError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    action = "verified" if args.check else "built"
    print(f"catalog {action}: {CATALOG_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
