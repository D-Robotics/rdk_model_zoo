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
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# Samples live on develop; the Web branch does not have to carry their source. When a sample
# directory is absent from this checkout, it is verified against this git ref instead.
SAMPLE_REF_ENV = "MODEL_ZOO_SAMPLE_REF"
SAMPLE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
BUILD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SIZE_ORDER = {name: index for index, name in enumerate(("n", "s", "m", "l", "x"))}
PLATFORM_ORDER = {name: index for index, name in enumerate(("x5", "s100", "s100p", "s600"))}
KNOWN_PLATFORMS = set(PLATFORM_ORDER)
ACCURACY_SCHEMA_PATH = WEB_ROOT / "accuracy-metrics.json"
ACCURACY_SCHEMA = json.loads(ACCURACY_SCHEMA_PATH.read_text(encoding="utf-8"))
TASK_REGISTRY_PATH = WEB_ROOT / "task-registry.json"
TASK_REGISTRY = json.loads(TASK_REGISTRY_PATH.read_text(encoding="utf-8"))
TASK_FACET_IDS = {item["facet_id"] for item in TASK_REGISTRY["tasks"].values()}
# Pending records stay out of the released-artifact schema. Sourced facts use
# the names below (`sourced_benchmarks`, `documented_downloads`, `example_image`),
# each tied to a file at sample_revision. The released names stay forbidden so a
# placeholder cannot be mistaken for a published release.
PENDING_FORBIDDEN = (
    "variants", "accuracy", "performance", "artifact", "assets", "benchmarks", "cover",
)
OFFICIAL_LICENSE_URLS = {
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "MIT": "https://opensource.org/license/mit",
    "BSD-2-Clause": "https://opensource.org/license/bsd-2-clause",
    "BSD-3-Clause": "https://opensource.org/license/bsd-3-clause",
    "GPL-2.0": "https://www.gnu.org/licenses/old-licenses/gpl-2.0.html",
    "GPL-3.0": "https://www.gnu.org/licenses/gpl-3.0.html",
    "LGPL-2.1": "https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html",
    "LGPL-3.0": "https://www.gnu.org/licenses/lgpl-3.0.html",
}
LICENSE_NAME_MARKERS = {
    "Apache-2.0": ("apache-2.0", "apache license, version 2.0", "apache license version 2.0"),
    "MIT": ("mit license",),
    "BSD-2-Clause": ("bsd 2-clause", "bsd-2-clause"),
    "BSD-3-Clause": ("bsd 3-clause", "bsd-3-clause"),
    "GPL-2.0": ("gpl-2.0", "gnu general public license version 2"),
    "GPL-3.0": ("gpl-3.0", "gnu general public license version 3"),
    "LGPL-2.1": ("lgpl-2.1",),
    "LGPL-3.0": ("lgpl-3.0",),
}
_BLOB_TEXT: dict[str, str] = {}


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


def _safe_samples_path(value: Any, label: str) -> str:
    sample_path = require_string(value, label)
    pure_path = PurePosixPath(sample_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or not pure_path.parts or pure_path.parts[0] != "samples":
        raise CatalogError(f"{label} must be a safe path below samples/")
    return sample_path


def _git_ref() -> str:
    ref = os.environ.get(SAMPLE_REF_ENV, "")
    if ref and not SAMPLE_REF_RE.match(ref):
        raise CatalogError(f"{SAMPLE_REF_ENV} is not a plain git ref: {ref!r}")
    return ref


def _ls_tree(ref: str, relative_path: str) -> tuple[str, str] | None:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", ref, "--", relative_path],
        capture_output=True, text=True, check=False,
    )
    if listed.returncode != 0:
        return None
    lines = [line for line in listed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    meta, _, listed_path = lines[0].partition("\t")
    if listed_path != relative_path:
        return None
    mode, object_type, _object_id = meta.split()
    return mode, object_type


def lookup_sample_entry(sample_path: str, label: str, revision: str | None = None) -> str:
    """Resolve pinned samples only in their commit; unpinned records retain the local/ref fallback."""
    if revision is not None:
        found = _ls_tree(revision, sample_path)
        if found == ("040000", "tree"):
            return "directory"
        if found == ("160000", "commit"):
            return "gitlink"
        raise CatalogError(f"{label} does not exist at pinned revision {revision}: {sample_path}")
    if (ROOT / sample_path).is_dir():
        return "directory"
    ref = _git_ref()
    if not ref:
        raise CatalogError(
            f"{label} does not exist: {sample_path} "
            f"(set {SAMPLE_REF_ENV} to a git ref such as origin/develop to verify it there)"
        )
    found = _ls_tree(ref, sample_path)
    if found == ("040000", "tree"):
        return "directory"
    if found == ("160000", "commit"):
        return "gitlink"
    raise CatalogError(f"{label} does not exist locally or in {ref}: {sample_path}")


def validate_sample_path(value: Any, label: str) -> str:
    sample_path = _safe_samples_path(value, label)
    if lookup_sample_entry(sample_path, label) != "directory":
        raise CatalogError(f"{label} is not a sample directory: {sample_path}")
    return sample_path


def validate_evidence_path(value: Any, label: str, revision: str | None = None) -> str:
    evidence = require_string(value, label)
    pure_path = PurePosixPath(evidence)
    if pure_path.is_absolute() or ".." in pure_path.parts or len(pure_path.parts) < 2:
        raise CatalogError(f"{label} must be a safe repository path")
    if not (evidence.startswith("docs/release/") or evidence.startswith("samples/")):
        raise CatalogError(f"{label} must point at docs/release/ or samples/")
    if revision is not None:
        found = _ls_tree(revision, evidence)
        if found is None or found[1] not in {"blob", "tree"}:
            raise CatalogError(f"{label} does not exist at pinned revision {revision}: {evidence}")
        return evidence
    if (ROOT / evidence).exists():
        return evidence
    ref = _git_ref()
    if not ref:
        raise CatalogError(
            f"{label} does not exist: {evidence} "
            f"(set {SAMPLE_REF_ENV} to a git ref such as origin/develop to verify it there)"
        )
    found = _ls_tree(ref, evidence)
    if found is None or found[1] not in {"blob", "tree"}:
        raise CatalogError(f"{label} does not exist locally or in {ref}: {evidence}")
    return evidence


def blob_text(relative_path: str, label: str, revision: str) -> str:
    """Text of a blob at the record's pinned revision, not the working tree."""
    cached = _BLOB_TEXT.get((revision, relative_path))
    if cached is not None:
        return cached
    found = _ls_tree(revision, relative_path)
    if found is None or found[1] != "blob":
        raise CatalogError(f"{label} is not a file at {revision}: {relative_path}")
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{revision}:{relative_path}"],
        capture_output=True, text=True, check=False,
    )
    if shown.returncode != 0:
        raise CatalogError(f"{label} could not be read from {revision}: {relative_path}")
    _BLOB_TEXT[(revision, relative_path)] = shown.stdout
    return shown.stdout


def _window_around(text: str, marker: str, span: int = 8000) -> str:
    index = text.find(marker)
    if index < 0:
        raise CatalogError(f"marker {marker!r} is not in the source file")
    window = text[index:index + span]
    # A benchmark id can sit a few kilobytes before the next list item. Do not
    # treat the next item's numbers as evidence for this one.
    boundary = window.find("\n  - id:", 1)
    if boundary > 0:
        window = window[:boundary]
    return window


def _require_in_text(needle: str, text: str, label: str) -> None:
    if needle not in text:
        raise CatalogError(f"{label} value {needle!r} is not in the source file")


def _number_in_text(value: Any, text: str, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{label} must be a number copied from the source file")
    if isinstance(value, float) and value.is_integer():
        candidates = {str(int(value)), str(value)}
    elif isinstance(value, float):
        compact = format(value, "f").rstrip("0").rstrip(".")
        candidates = {compact, str(value)}
    else:
        candidates = {str(value)}
    if not any(candidate in text for candidate in candidates):
        raise CatalogError(f"{label} value {value!r} is not in the source file")


def _require_https(value: Any, label: str) -> str:
    url = require_string(value, label)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise CatalogError(f"{label} must be an absolute HTTPS URL")
    return url


def validate_commit(value: Any, label: str) -> str:
    revision = require_string(value, label)
    if not COMMIT_RE.fullmatch(revision):
        raise CatalogError(f"{label} must be a full 40-character commit SHA")
    kind = subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-t", revision],
        capture_output=True, text=True, check=False,
    )
    if kind.returncode != 0 or kind.stdout.strip() != "commit":
        raise CatalogError(f"{label} is not a commit in this repository: {revision}")
    return revision


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


def _validate_pending_facts(record: dict[str, Any], relative_source: str) -> None:
    """Every optional fact must be a substring of the file it cites at sample_revision."""
    revision = require_string(record.get("sample_revision"), f"{relative_source}.sample_revision")
    def text_of(path: str, label: str) -> str:
        return blob_text(path, label, revision)
    upstream = record.get("upstream", [])
    if not isinstance(upstream, list):
        raise CatalogError(f"{relative_source}.upstream must be a list")
    seen_upstream: set[str] = set()
    for index, item in enumerate(upstream):
        label = f"{relative_source}.upstream[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"kind", "name", "url", "source"}:
            raise CatalogError(f"{label} has unknown fields")
        if entry.get("kind") not in {"repository", "paper", "model-card"}:
            raise CatalogError(f"{label}.kind must be repository, paper, or model-card")
        require_string(entry.get("name"), f"{label}.name")
        url = _require_https(entry.get("url"), f"{label}.url")
        resource_path = urlparse(url).path.rstrip("/")
        if entry["kind"] == "repository" and (
            "/releases/download/" in resource_path
            or re.search(r"\.(?:pt|pth|onnx|bin|hbm|zip|tar\.gz)$", resource_path, re.IGNORECASE)
        ):
            raise CatalogError(f"{label}: repository links must not be weight or archive downloads")
        canonical_url = url.rstrip("/").removesuffix(".git")
        if canonical_url in seen_upstream:
            raise CatalogError(f"{label}: duplicate upstream URL {canonical_url}")
        seen_upstream.add(canonical_url)
        source = validate_evidence_path(entry.get("source"), f"{label}.source", revision)
        _require_in_text(url, text_of(source, f"{label}.source"), label)

    variants = record.get("published_variants", [])
    if not isinstance(variants, list):
        raise CatalogError(f"{relative_source}.published_variants must be a list")
    for index, item in enumerate(variants):
        label = f"{relative_source}.published_variants[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"name", "source"}:
            raise CatalogError(f"{label} has unknown fields")
        name = require_string(entry.get("name"), f"{label}.name")
        source = validate_evidence_path(entry.get("source"), f"{label}.source", revision)
        _require_in_text(name, text_of(source, f"{label}.source"), label)

    configs = record.get("conversion_configs", [])
    if not isinstance(configs, list):
        raise CatalogError(f"{relative_source}.conversion_configs must be a list")
    for index, item in enumerate(configs):
        label = f"{relative_source}.conversion_configs[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"path", "march", "runtime_input", "train_input", "train_layout", "resolution"}:
            raise CatalogError(f"{label} has unknown fields")
        path = validate_evidence_path(entry.get("path"), f"{label}.path", revision)
        text = text_of(path, f"{label}.path")
        for field in ("march", "runtime_input", "train_input", "train_layout"):
            if field in entry:
                _require_in_text(require_string(entry[field], f"{label}.{field}"), text, f"{label}.{field}")
        if "resolution" in entry:
            resolution = require_string(entry["resolution"], f"{label}.resolution")
            if resolution not in path and resolution not in text:
                raise CatalogError(f"{label}.resolution is not in the config path or file")

    inputs = record.get("inputs", [])
    if not isinstance(inputs, list):
        raise CatalogError(f"{relative_source}.inputs must be a list")
    for index, item in enumerate(inputs):
        label = f"{relative_source}.inputs[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"platform", "shape", "layout", "format", "source", "source_id"}:
            raise CatalogError(f"{label} has unknown fields")
        platform = require_string(entry.get("platform"), f"{label}.platform")
        if platform not in KNOWN_PLATFORMS:
            raise CatalogError(f"{label}.platform must be one of {', '.join(sorted(KNOWN_PLATFORMS))}")
        source = validate_evidence_path(entry.get("source"), f"{label}.source", revision)
        source_id = require_string(entry.get("source_id"), f"{label}.source_id")
        window = _window_around(text_of(source, f"{label}.source"), source_id)
        for field in ("shape", "layout", "format"):
            if field in entry:
                _require_in_text(require_string(entry[field], f"{label}.{field}"), window, f"{label}.{field}")
        if not any(field in entry for field in ("shape", "layout", "format")):
            raise CatalogError(f"{label} must record a shape, layout, or format")

    usage = record.get("usage", [])
    if not isinstance(usage, list):
        raise CatalogError(f"{relative_source}.usage must be a list")
    for index, item in enumerate(usage):
        label = f"{relative_source}.usage[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"source", "lines"}:
            raise CatalogError(f"{label} has unknown fields")
        source = validate_evidence_path(entry.get("source"), f"{label}.source", revision)
        text = text_of(source, f"{label}.source")
        lines = require_list(entry.get("lines"), f"{label}.lines")
        for line_index, line in enumerate(lines):
            command = require_string(line, f"{label}.lines[{line_index}]")
            _require_in_text(command, text, f"{label}.lines[{line_index}]")

    measurements = record.get("sourced_benchmarks", [])
    if not isinstance(measurements, list):
        raise CatalogError(f"{relative_source}.sourced_benchmarks must be a list")
    for index, item in enumerate(measurements):
        label = f"{relative_source}.sourced_benchmarks[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"id", "platform", "variant", "metric", "value", "unit", "statistic", "condition", "source", "origin"}:
            raise CatalogError(f"{label} has unknown fields")
        platform = require_string(entry.get("platform"), f"{label}.platform")
        if platform not in KNOWN_PLATFORMS:
            raise CatalogError(f"{label}.platform must be one of {', '.join(sorted(KNOWN_PLATFORMS))}")
        metric = require_string(entry.get("metric"), f"{label}.metric")
        unit = require_string(entry.get("unit"), f"{label}.unit")
        source_id = require_string(entry.get("id"), f"{label}.id")
        source = validate_evidence_path(entry.get("source"), f"{label}.source", revision)
        window = _window_around(text_of(source, f"{label}.source"), source_id, span=20000)
        _require_in_text(metric, window, f"{label}.metric")
        _require_in_text(unit, window, f"{label}.unit")
        _number_in_text(entry.get("value"), window, f"{label}.value")
        for field in ("variant", "statistic", "condition"):
            if field in entry:
                _require_in_text(require_string(entry[field], f"{label}.{field}"), window, f"{label}.{field}")
        if "origin" in entry:
            validate_evidence_path(entry.get("origin"), f"{label}.origin", revision)

    downloads = record.get("documented_downloads", [])
    if not isinstance(downloads, list):
        raise CatalogError(f"{relative_source}.documented_downloads must be a list")
    for index, item in enumerate(downloads):
        label = f"{relative_source}.documented_downloads[{index}]"
        entry = require_mapping(item, label)
        if set(entry) - {"platform", "filename", "url", "format", "sha256", "source"}:
            raise CatalogError(f"{label} has unknown fields")
        platform = require_string(entry.get("platform"), f"{label}.platform")
        if platform not in KNOWN_PLATFORMS:
            raise CatalogError(f"{label}.platform must be one of {', '.join(sorted(KNOWN_PLATFORMS))}")
        filename = require_string(entry.get("filename"), f"{label}.filename")
        url = _require_https(entry.get("url"), f"{label}.url")
        source = validate_evidence_path(entry.get("source"), f"{label}.source", revision)
        text = text_of(source, f"{label}.source")
        _require_in_text(filename, text, f"{label}.filename")
        _require_in_text(url, text, f"{label}.url")
        if "format" in entry:
            _require_in_text(require_string(entry["format"], f"{label}.format"), text, f"{label}.format")
        if "sha256" in entry:
            digest = require_string(entry.get("sha256"), f"{label}.sha256")
            if not SHA256_RE.fullmatch(digest):
                raise CatalogError(f"{label}.sha256 must be 64 lowercase hexadecimal characters")
            _require_in_text(digest, text, f"{label}.sha256")

    if "example_image" in record:
        image = require_mapping(record.get("example_image"), f"{relative_source}.example_image")
        if set(image) - {"path", "source"}:
            raise CatalogError(f"{relative_source}.example_image has unknown fields")
        path = validate_evidence_path(image.get("path"), f"{relative_source}.example_image.path", revision)
        source = validate_evidence_path(image.get("source"), f"{relative_source}.example_image.source", revision)
        if path != source:
            raise CatalogError(f"{relative_source}.example_image.path and source must be the same blob")
        suffix = PurePosixPath(path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            raise CatalogError(f"{relative_source}.example_image.path must be an image")
        found = _ls_tree(revision, path)
        if found is None or found[1] != "blob":
            raise CatalogError(f"{relative_source}.example_image.path is not a file at {revision}: {path}")

    if "source_input" in record:
        entry = require_mapping(record.get("source_input"), f"{relative_source}.source_input")
        if set(entry) - {"format", "dtype", "layout", "shape", "source"}:
            raise CatalogError(f"{relative_source}.source_input has unknown fields")
        source = validate_evidence_path(entry.get("source"), f"{relative_source}.source_input.source", revision)
        text = text_of(source, f"{relative_source}.source_input.source")
        present = [field for field in ("format", "dtype", "layout", "shape") if field in entry]
        if not present:
            raise CatalogError(f"{relative_source}.source_input needs a sourced format, dtype, layout, or shape")
        for field in present:
            _require_in_text(require_string(entry[field], f"{relative_source}.source_input.{field}"), text, f"{relative_source}.source_input.{field}")

    if "hero_visual" in record:
        _validate_hero_visual(record, relative_source, revision, text_of)


def _hero_asset_path(value: Any, label: str, suffix: str) -> str:
    hero_file = require_string(value, label)
    hero_path = PurePosixPath(hero_file)
    if hero_path.is_absolute() or ".." in hero_path.parts or hero_path.parts[:4] != (
        "model_zoo_web", "release", "assets", "pending-heroes",
    ) or hero_path.suffix.lower() != suffix:
        raise CatalogError(f"{label} must be a {suffix} file under model_zoo_web/release/assets/pending-heroes/")
    if not (ROOT / hero_file).is_file():
        raise CatalogError(f"{label} is missing: {hero_file}")
    return hero_file


def _validate_host_float_hero(entry: dict[str, Any], label: str, revision: str) -> None:
    """Predictions come from a host run of the documented float checkpoint, stored beside the hero."""
    note = require_string(entry.get("note"), f"{label}.note")
    if "not board output" not in note or "Host float reference" not in note:
        raise CatalogError(f"{label}.note must say this is a host float reference, not board output")
    if require_string(entry.get("commit"), f"{label}.commit") != revision:
        raise CatalogError(f"{label}.commit must be the pinned sample revision")
    model_url = require_string(entry.get("model_url"), f"{label}.model_url")
    if not model_url.startswith("https://"):
        raise CatalogError(f"{label}.model_url must be an https checkpoint URL")
    require_string(entry.get("model_file"), f"{label}.model_file")
    require_string(entry.get("framework"), f"{label}.framework")
    documented = validate_evidence_path(entry.get("documented_in"), f"{label}.documented_in", revision)
    found = _ls_tree(revision, documented)
    if found is None or found[1] != "blob":
        raise CatalogError(f"{label}.documented_in is not a file at {revision}: {documented}")
    source = _hero_asset_path(entry.get("predictions_source"), f"{label}.predictions_source", ".json")
    payload = json.loads((ROOT / source).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CatalogError(f"{label}.predictions_source must be a JSON object")
    for key in ("model_url", "model_file", "input", "framework", "commit", "runtime"):
        if payload.get(key) != entry.get(key):
            raise CatalogError(f"{label}.{key} does not match the inference record")
    if payload.get("runtime") != "host-float-reference":
        raise CatalogError(f"{label} inference record must be host-float-reference")
    kind = entry["kind"]
    if kind == "classification-panel":
        recorded = payload.get("topk")
        declared = entry.get("topk")
        if not isinstance(recorded, list) or len(recorded) != 5:
            raise CatalogError(f"{label} inference record needs five Top-K rows")
        if recorded != declared:
            raise CatalogError(f"{label}.topk must be the five recorded probabilities")
        for index, row in enumerate(recorded):
            if not isinstance(row, dict):
                raise CatalogError(f"{label}.topk[{index}] must be an object")
            require_string(row.get("label"), f"{label}.topk[{index}].label")
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
                raise CatalogError(f"{label}.topk[{index}].score must be a probability")
        return
    if kind not in {"detection-overlay", "segmentation-overlay"}:
        raise CatalogError(f"{label}.kind is not a host-float hero")
    recorded = payload.get("detections")
    declared = entry.get("detections")
    if not isinstance(recorded, list) or not recorded:
        raise CatalogError(f"{label} inference record needs at least one detection")
    if recorded != declared:
        raise CatalogError(f"{label}.detections must be the recorded boxes")
    for index, row in enumerate(recorded):
        if not isinstance(row, dict):
            raise CatalogError(f"{label}.detections[{index}] must be an object")
        require_string(row.get("label"), f"{label}.detections[{index}].label")
        score = row.get("score")
        box = row.get("box")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= float(score) <= 1:
            raise CatalogError(f"{label}.detections[{index}].score must be a probability")
        if (
            not isinstance(box, list) or len(box) != 4
            or not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in box)
        ):
            raise CatalogError(f"{label}.detections[{index}].box must be four numbers")
    threshold = entry.get("confidence_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < float(threshold) <= 1:
        raise CatalogError(f"{label}.confidence_threshold must be a probability cutoff")
    if payload.get("confidence_threshold") != threshold:
        raise CatalogError(f"{label}.confidence_threshold does not match the inference record")
    for index, row in enumerate(recorded):
        if float(row["score"]) < float(threshold):
            raise CatalogError(f"{label}.detections[{index}].score is below the confidence threshold")
    if kind == "segmentation-overlay" and not payload.get("masks"):
        raise CatalogError(f"{label} segmentation record must say masks were drawn")


def _validate_hero_visual(record: dict[str, Any], relative_source: str, revision: str, text_of) -> None:
    """A pending hero is a host float render or a crop/quote taken from a pinned file."""
    label = f"{relative_source}.hero_visual"
    entry = require_mapping(record.get("hero_visual"), label)
    allowed = {
        "file", "kind", "input", "predictions_source", "runtime", "quote", "labels", "scores",
        "crop", "note", "model_file", "model_url", "framework", "commit", "documented_in",
        "topk", "detections", "confidence_threshold",
    }
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise CatalogError(f"{label} has unknown fields: {', '.join(unknown)}")
    kind = require_string(entry.get("kind"), f"{label}.kind")
    image = validate_evidence_path(entry.get("input"), f"{label}.input", revision)
    suffix = PurePosixPath(image).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise CatalogError(f"{label}.input must be an image")
    found = _ls_tree(revision, image)
    if found is None or found[1] != "blob":
        raise CatalogError(f"{label}.input is not a file at {revision}: {image}")
    _hero_asset_path(entry.get("file"), f"{label}.file", ".webp")
    runtime = entry.get("runtime")
    if runtime == "host-float-reference":
        _validate_host_float_hero(entry, label, revision)
        return
    if runtime != "display-composition":
        raise CatalogError(f"{label}.runtime must be display-composition or host-float-reference")
    if kind not in {"classification-panel", "result-crop"}:
        raise CatalogError(f"{label}.kind must be classification-panel or result-crop")
    source = validate_evidence_path(entry.get("predictions_source"), f"{label}.predictions_source", revision)
    if kind == "result-crop":
        if source != image:
            raise CatalogError(f"{label}.predictions_source must be the cropped result image")
        if "labels" in entry or "scores" in entry or "quote" in entry:
            raise CatalogError(f"{label} result crop must not add labels")
        crop = entry.get("crop")
        if not isinstance(crop, list) or len(crop) != 4 or not all(isinstance(item, int) and item >= 0 for item in crop):
            raise CatalogError(f"{label}.crop must be four non-negative integers")
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise CatalogError(f"{label}.crop must be a non-empty box")
        return
    quote = require_string(entry.get("quote"), f"{label}.quote")
    _require_in_text(quote, text_of(source, f"{label}.predictions_source"), f"{label}.quote")
    labels = entry.get("labels")
    if not isinstance(labels, list) or not labels:
        raise CatalogError(f"{label}.labels must list the quoted classes")
    cursor = 0
    for index, item in enumerate(labels):
        text = require_string(item, f"{label}.labels[{index}]")
        found_at = quote.find(text, cursor)
        if found_at < 0:
            raise CatalogError(f"{label}.labels[{index}] is not in the quote in order")
        cursor = found_at + len(text)
    scores = entry.get("scores", [])
    if scores is None:
        scores = []
    if not isinstance(scores, list):
        raise CatalogError(f"{label}.scores must be a list")
    if scores and len(scores) != len(labels):
        raise CatalogError(f"{label}.scores must align with labels")
    for index, score in enumerate(scores):
        score_text = require_string(score, f"{label}.scores[{index}]")
        if score_text not in quote:
            raise CatalogError(f"{label}.scores[{index}] is not in the quote")
        try:
            value = float(score_text)
        except ValueError as exc:
            raise CatalogError(f"{label}.scores[{index}] must be a decimal copied from the quote") from exc
        if not 0 <= value <= 1:
            raise CatalogError(f"{label}.scores[{index}] must be a ratio between 0 and 1")


def validate_pending_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Identity plus sourced facts. Released artifacts and unsourced measurements are refused."""
    relative_source = source_path.relative_to(ROOT).as_posix()
    allowed = {
        "schema_version", "registration", "id", "name", "domain", "source", "family", "task",
        "description", "sample_path", "sample_revision", "sample_kind", "supported_platforms",
        "facet_tasks", "manifest_tasks", "license", "provider",
        "upstream", "published_variants", "conversion_configs", "inputs", "usage",
        "sourced_benchmarks", "documented_downloads", "example_image", "source_input",
        "hero_visual",
    }
    for field in PENDING_FORBIDDEN:
        if field in record:
            raise CatalogError(
                f"{relative_source}: pending records must not include {field}; "
                "remove registration: pending when the released record is complete"
            )
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise CatalogError(f"{relative_source}: unknown pending fields: {', '.join(unknown)}")
    record_id = require_string(record.get("id"), f"{relative_source}.id")
    domain = require_string(record.get("domain"), f"{relative_source}.domain")
    source = require_string(record.get("source"), f"{relative_source}.source")
    family = require_string(record.get("family"), f"{relative_source}.family")
    task = require_string(record.get("task"), f"{relative_source}.task")
    require_string(record.get("name"), f"{relative_source}.name")
    if task not in TASK_REGISTRY["tasks"]:
        known = ", ".join(sorted(TASK_REGISTRY["tasks"]))
        raise CatalogError(f"{relative_source}.task must be one of {known}")
    validate_localized_text(record.get("description"), f"{relative_source}.description")
    expected_id = f"{source}/{family}/{task}"
    if record_id != expected_id:
        raise CatalogError(f"{relative_source}.id must be {expected_id}")
    expected_source = DATA_ROOT / domain / source / family / f"{task}.yaml"
    if source_path != expected_source:
        raise CatalogError(f"{relative_source}: expected path {expected_source.relative_to(ROOT)}")

    revision = validate_commit(record.get("sample_revision"), f"{relative_source}.sample_revision")
    sample_path = _safe_samples_path(record.get("sample_path"), f"{relative_source}.sample_path")
    entry_kind = lookup_sample_entry(sample_path, f"{relative_source}.sample_path", revision)
    declared_kind = record.get("sample_kind", "directory")
    if declared_kind not in {"directory", "gitlink"}:
        raise CatalogError(f"{relative_source}.sample_kind must be directory or gitlink")
    if declared_kind != entry_kind:
        raise CatalogError(
            f"{relative_source}.sample_kind is {declared_kind} but {sample_path} is a {entry_kind}"
        )

    if "license" in record:
        license_info = require_mapping(record.get("license"), f"{relative_source}.license")
        license_name = require_string(license_info.get("name"), f"{relative_source}.license.name")
        license_url = _require_https(license_info.get("url"), f"{relative_source}.license.url")
        unknown_license_keys = set(license_info) - {"name", "url", "source", "scope"}
        if unknown_license_keys:
            raise CatalogError(f"{relative_source}.license has unknown fields: {sorted(unknown_license_keys)}")
        if "scope" in license_info and license_info["scope"] not in {"sample-code", "model"}:
            raise CatalogError(f"{relative_source}.license.scope must be sample-code or model")
        if "source" in license_info:
            license_source = validate_evidence_path(license_info.get("source"), f"{relative_source}.license.source", revision)
            license_text = blob_text(
                license_source, f"{relative_source}.license.source", record["sample_revision"],
            ).lower()
            markers = LICENSE_NAME_MARKERS.get(license_name, (license_name.lower(),))
            if not any(marker in license_text for marker in markers):
                raise CatalogError(f"{relative_source}.license.name is not named in {license_source}")
            official = OFFICIAL_LICENSE_URLS.get(license_name)
            if license_url not in license_text and license_url != official:
                raise CatalogError(f"{relative_source}.license.url is neither in {license_source} nor the official page for {license_name}")
        elif license_name in OFFICIAL_LICENSE_URLS and license_url != OFFICIAL_LICENSE_URLS[license_name]:
            raise CatalogError(f"{relative_source}.license.url must be the official page when no source file is cited")
    if "provider" in record:
        require_string(record.get("provider"), f"{relative_source}.provider")
    _validate_pending_facts(record, relative_source)

    platforms = require_list(record.get("supported_platforms"), f"{relative_source}.supported_platforms")
    seen_platforms: set[str] = set()
    for index, platform_value in enumerate(platforms):
        platform_label = f"{relative_source}.supported_platforms[{index}]"
        platform_record = require_mapping(platform_value, platform_label)
        platform = require_string(platform_record.get("platform"), f"{platform_label}.platform")
        if platform not in KNOWN_PLATFORMS:
            raise CatalogError(f"{platform_label}.platform must be one of {', '.join(sorted(KNOWN_PLATFORMS))}")
        if platform in seen_platforms:
            raise CatalogError(f"{relative_source}: duplicate supported platform {platform}")
        seen_platforms.add(platform)
        validate_evidence_path(platform_record.get("evidence"), f"{platform_label}.evidence", revision)

    facet_tasks = record.get("facet_tasks", [TASK_REGISTRY["tasks"][task]["facet_id"]])
    if not isinstance(facet_tasks, list) or not facet_tasks:
        raise CatalogError(f"{relative_source}.facet_tasks must be a non-empty list")
    for index, facet_id in enumerate(facet_tasks):
        if facet_id not in TASK_FACET_IDS:
            raise CatalogError(f"{relative_source}.facet_tasks[{index}] is not a registered facet")
    manifest_tasks = record.get("manifest_tasks", [])
    if not isinstance(manifest_tasks, list) or any(not isinstance(item, str) or not item.strip() for item in manifest_tasks):
        raise CatalogError(f"{relative_source}.manifest_tasks must be a list of strings")

    normalized = copy.deepcopy(record)
    normalized["source_file"] = relative_source
    normalized["variants"] = []
    normalized["supported_platforms"].sort(key=lambda item: PLATFORM_ORDER[item["platform"]])
    return normalized


def validate_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    relative_source = source_path.relative_to(ROOT).as_posix()
    if record.get("schema_version") != 1:
        raise CatalogError(f"{relative_source}: schema_version must be 1")
    registration = record.get("registration")
    if registration == "pending":
        return validate_pending_record(record, source_path)
    if registration is not None:
        raise CatalogError(f"{relative_source}.registration must be pending or omitted")

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
    pending_count = 0
    for source_path in source_paths:
        with source_path.open(encoding="utf-8") as handle:
            record = require_mapping(yaml.safe_load(handle), source_path.relative_to(ROOT).as_posix())
        normalized = validate_record(record, source_path)
        if normalized["id"] in seen_ids:
            raise CatalogError(f"duplicate model id: {normalized['id']}")
        seen_ids.add(normalized["id"])
        if normalized.get("registration") == "pending":
            pending_count += 1
        else:
            release_count += sum(len(variant["platforms"]) for variant in normalized["variants"])
        records.append(normalized)

    records.sort(key=lambda item: item["id"])
    catalog = {
        "schema_version": 1,
        "source": "model_zoo_web/data",
        "model_count": len(records),
        "release_count": release_count,
        "pending_count": pending_count,
        "models": records,
    }
    catalog_bytes = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode()
    catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    metadata = {
        "schema_version": 1,
        "catalog": "catalog.json",
        "sha256": catalog_sha256,
        "bytes": len(catalog_bytes),
        "model_count": len(records),
        "release_count": release_count,
        "pending_count": pending_count,
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
