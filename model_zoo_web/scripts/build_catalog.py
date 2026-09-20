#!/usr/bin/env python3
"""Build the samples-only Model Zoo Web catalog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
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
SIZE_ORDER = {name: index for index, name in enumerate(("n", "s", "m", "l", "x"))}


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


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogError(f"{label} must be a non-negative integer")
    return value


def validate_sample_path(value: Any, label: str) -> str:
    sample_path = require_string(value, label)
    pure_path = PurePosixPath(sample_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.parts[0] != "samples":
        raise CatalogError(f"{label} must be a safe path below samples/")
    if not (ROOT / sample_path).is_dir():
        raise CatalogError(f"{label} does not exist: {sample_path}")
    return sample_path


def validate_artifact(
    artifact: dict[str, Any],
    *,
    source: str,
    family: str,
    task: str,
    size: str,
    platform: str,
    label: str,
) -> None:
    url = require_string(artifact.get("url"), f"{label}.url")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != OSS_HOST or parsed.query or parsed.fragment:
        raise CatalogError(f"{label}.url must be a direct HTTPS URL on {OSS_HOST}")
    filename = PurePosixPath(unquote(parsed.path)).name
    expected_path = f"/models/{source}/{family}/{task}/{size}/{platform}/{filename}"
    if unquote(parsed.path) != expected_path:
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
    require_positive_int(
        end_to_end.get("cpu_frequency_mhz"),
        f"{label}.cpu_frequency_mhz",
    )
    require_positive_int(
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


def validate_record(record: dict[str, Any], source_path: Path) -> dict[str, Any]:
    relative_source = source_path.relative_to(ROOT).as_posix()
    if record.get("schema_version") != 1:
        raise CatalogError(f"{relative_source}: schema_version must be 1")

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
            )

            reports = require_mapping(platform_record.get("reports"), f"{platform_label}.reports")
            release_base_url = artifact["url"].rsplit("/", 1)[0]
            expected_report_url = f"{release_base_url}/oe_report.html"
            if reports.get("oe_conversion_url") != expected_report_url:
                raise CatalogError(
                    f"{platform_label}.reports.oe_conversion_url must be {expected_report_url}"
                )

            accuracy = require_mapping(platform_record.get("accuracy"), f"{platform_label}.accuracy")
            runtime_accuracy = require_mapping(accuracy.get("runtime"), f"{platform_label}.accuracy.runtime")
            require_number(runtime_accuracy.get("map_50_95"), f"{platform_label}.accuracy.runtime.map_50_95")
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
    normalized["variants"].sort(key=lambda item: (SIZE_ORDER.get(item["size"], 99), item["size"]))
    for variant in normalized["variants"]:
        variant["platforms"].sort(key=lambda item: item["platform"])
    return normalized


def build_catalog() -> tuple[bytes, bytes]:
    source_paths = sorted(DATA_ROOT.rglob("*.yaml"))
    if not source_paths:
        raise CatalogError(f"no YAML records found below {DATA_ROOT}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    release_count = 0
    for source_path in source_paths:
        with source_path.open(encoding="utf-8") as handle:
            record = require_mapping(yaml.safe_load(handle), source_path.relative_to(ROOT).as_posix())
        normalized = validate_record(record, source_path)
        if normalized["id"] in seen_ids:
            raise CatalogError(f"duplicate model id: {normalized['id']}")
        seen_ids.add(normalized["id"])
        release_count += sum(len(variant["platforms"]) for variant in normalized["variants"])
        records.append(normalized)

    records.sort(key=lambda item: item["id"])
    catalog = {
        "schema_version": 1,
        "source": "model_zoo_web/data",
        "model_count": len(records),
        "release_count": release_count,
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
