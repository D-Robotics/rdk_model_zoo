#!/usr/bin/env python3
"""Add verified X5 task releases to the existing model catalog.

By default the plan selects all twenty task-size variants; --models can select
a fully verified subset for incremental promotion. The default is read-only.
No pending package or unmeasured performance may become an official entry,
and existing platform records remain intact.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable

import yaml

from promote_yolo26_task_batch import (
    _load_catalog_builder, _make_platform_record,
    _verify_accuracy, _verify_sums, need, read_json, safe_regular, sha256,
)

WEB_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("cls", "seg", "pose", "obb")
SIZES = ("n", "s", "m", "l", "x")
DEFAULT_WORKBENCH = Path("/home/zhengyi/work/rdk_model_zoo_workbench")
ALL_MODELS = tuple((task, size) for task in TASKS for size in SIZES)


def parse_models(value: str) -> tuple[tuple[str, str], ...]:
    """Parse comma-separated model names in task-size form."""
    if not value.strip():
        raise argparse.ArgumentTypeError("--models must select at least one task-size")
    selected = []
    seen = set()
    choices = ", ".join(f"{task}-{size}" for task, size in ALL_MODELS)
    for raw_item in value.split(","):
        item = raw_item.strip()
        parts = item.split("-")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid model {item!r}; expected one of {choices}")
        task, size = parts
        if task not in TASKS or size not in SIZES:
            raise argparse.ArgumentTypeError(
                f"Invalid model {item!r}; expected one of {choices}")
        model = (task, size)
        if model in seen:
            raise argparse.ArgumentTypeError(f"Duplicate model selection: {item}")
        seen.add(model)
        selected.append(model)
    return tuple(selected)


def normalize_models(models: tuple[tuple[str, str], ...] | None) -> tuple[tuple[str, str], ...]:
    if models is None:
        return ALL_MODELS
    selected = tuple(models)
    need(bool(selected), "At least one model must be selected")
    seen = set()
    for model in selected:
        need(isinstance(model, (tuple, list)) and len(model) == 2,
             f"Invalid model selection: {model!r}")
        task, size = model
        need(task in TASKS and size in SIZES, f"Invalid model selection: {model!r}")
        need((task, size) not in seen, f"Duplicate model selection: {task}-{size}")
        seen.add((task, size))
    return tuple((task, size) for task, size in selected)


def normalize_performance_for_catalog(performance: dict) -> dict:
    """Copy measured X5 timing data and fill only provable per-round frame counts."""
    normalized = copy.deepcopy(performance)
    end_to_end = normalized.get("end_to_end")
    records = [end_to_end] if isinstance(end_to_end, dict) else end_to_end
    need(isinstance(records, list) and records, "X5 end-to-end measurements are missing")
    for index, record in enumerate(records):
        need(isinstance(record, dict), f"end_to_end[{index}] must be an object")
        if record.get("frames_per_round") is not None:
            continue
        required = ("pipeline_streams", "runs_per_round", "rounds", "timed_frames")
        for field in required:
            value = record.get(field)
            need(isinstance(value, int) and not isinstance(value, bool) and value > 0,
                 f"end_to_end[{index}].{field} must be a positive integer to derive frames_per_round")
        expected_frames = record["pipeline_streams"] * record["runs_per_round"] * record["rounds"]
        need(record["timed_frames"] == expected_frames,
             f"end_to_end[{index}].timed_frames does not match streams × runs_per_round × rounds")
        record["frames_per_round"] = record["runs_per_round"]
    return normalized


def load_workbench(workbench: Path):
    scripts = workbench / "scripts"
    need(scripts.is_dir(), "Missing Workbench scripts")
    sys.path.insert(0, str(scripts))
    import stage_x5_task_release
    import verify_task_release_publication
    need(stage_x5_task_release.ROOT.resolve() == workbench.resolve(), "Wrong Workbench audit root")
    need(verify_task_release_publication.ROOT.resolve() == workbench.resolve(), "Wrong publication verifier root")
    return stage_x5_task_release, verify_task_release_publication


def verified_entry(workbench: Path, task: str, size: str, release_date: str) -> tuple[dict, bytes]:
    gate, verifier = load_workbench(workbench)
    audit = gate.audit_one(task, size)
    need(audit.get("name") == f"{task}-{size}-x5" and audit.get("ready") is True,
         f"{task}/{size}: current X5 gate is not ready")
    checks = audit.get("checks")
    need(isinstance(checks, list) and len(checks) >= 11 and
         all(row.get("passed") is True for row in checks), f"{task}/{size}: incomplete X5 validation")
    stage = workbench / "outputs/analysis/x5-publication-20261001/staged" / task / size / "x5"
    staged = read_json(stage / "release.json")
    canonical = f"models/ultralytics_yolo/yolo26/{task}/{size}/x5"
    prefix = staged.get("object_prefix", canonical)
    build_id = staged["provenance"]["build_id"]
    need(prefix in {canonical, f"{canonical}/rebuilds/{build_id}"}, "Unexpected X5 object prefix")
    name = f"{task}-{size}-x5" + (f"-{build_id}" if prefix != canonical else "")
    finalization_path = workbench / "outputs/receipts/finalizations" / f"{name}.json"
    verified = verifier.verify_publication(finalization_path)
    finalization = read_json(finalization_path)
    need(finalization.get("evidence_sources") == audit["evidence_sources"] == staged.get("evidence_sources"),
         "Evidence changed after the X5 bundle was staged or finalized")
    need(finalization["stage"]["release_manifest_sha256"] == sha256(stage / "release.json") and
         finalization["stage"]["sha256sums_sha256"] == sha256(stage / "SHA256SUMS"),
         "Finalization does not bind the exact staged bundle")
    completion_path = finalization_path.with_suffix(".publication.json")
    completion = read_json(completion_path)
    need(completion.get("status") == "verified-published" and completion.get("verified_object_count") == 5 and
         completion.get("uploaded_objects") == verified["uploaded_objects"] and
         completion.get("finalization_receipt", {}).get("sha256") == sha256(finalization_path),
         "Missing or stale completed five-object publication receipt")
    bundle = Path(finalization["finalized_bundle"]["path"])
    release = read_json(bundle / "release.json")
    need(release.get("schema_version") == 2 and release.get("status") == "released" and
         release.get("publication_status") == "published", "Expected a finalized published schema-2 manifest")
    need(release.get("target") == {"platform": "x5", "march": "bayes-e", "format": "bin"} and
         release.get("model", {}).get("task") == task and release["model"].get("size") == size,
         "X5 target or model identity mismatch")
    _verify_sums(bundle, release["artifact"]["name"])
    context = audit["context"]
    need(context["model_sha256"] == release["artifact"]["sha256"], "Published BIN differs from current audit")
    campaign_path = Path(context["campaign_path"])
    need(campaign_path.resolve().is_relative_to(workbench) and
         sha256(campaign_path) == context["campaign_sha256"] == release["provenance"]["campaign_sha256"],
         "Campaign identity mismatch")
    campaign = read_json(campaign_path)
    accuracy = _verify_accuracy(task, release, campaign)
    performance = normalize_performance_for_catalog(release["performance"]["web_record"])
    need(performance.get("stages", {}).get("runtime") == "measured" and
         {m.get("threads") for m in performance.get("measurements", [])} == {1, 2} and
         {e.get("pipeline_streams") for e in performance.get("end_to_end", [])} == {1, 2},
         "Both native concurrency and C++ stream conditions must be measured")
    urls = {row["name"]: row["url"] for row in verified["uploaded_objects"].values()}
    # X5 manifests record the runtime source fingerprint under source_sha256.
    compatible_release = copy.deepcopy(release)
    compatible_release["provenance"].setdefault(
        "runtime_source_manifest_sha256", release["provenance"].get("runtime_source_sha256"))
    entry = _make_platform_record(task, size, "x5", campaign, compatible_release, {"urls": urls},
                                  {"accuracy": accuracy, "performance": performance}, release_date)
    if "runtime_source_manifest_sha256" not in release["provenance"]:
        entry["provenance"]["runtime_source_sha256"] = entry["provenance"].pop(
            "runtime_source_manifest_sha256")
    if entry["provenance"].get("runtime_source_commit") is None:
        entry["provenance"].pop("runtime_source_commit", None)
    return entry, (bundle / "oe_report_data.json").read_bytes()


def make_plan(workbench: Path, release_date: str,
              entry_provider: Callable | None = None,
              models: tuple[tuple[str, str], ...] | None = None,
              originals: dict[Path, bytes] | None = None) -> tuple[dict[Path, bytes], dict]:
    workbench = workbench.resolve()
    provider = entry_provider or verified_entry
    selected = normalize_models(models)
    selected_set = set(selected)
    inputs_path = WEB_ROOT / "release/inputs.json"
    safe_regular(inputs_path, "Existing release inputs")
    source_inputs_bytes = inputs_path.read_bytes()
    source_inputs = json.loads(source_inputs_bytes)
    if originals is not None:
        originals[inputs_path] = source_inputs_bytes
    source_releases = source_inputs.get("releases")
    need(source_inputs.get("schema_version") == 1 and isinstance(source_releases, dict) and
         len(source_releases) >= 100,
         "Expected a deployed catalog with at least 100 entries")
    inputs = copy.deepcopy(source_inputs)
    writes = {}
    builder = _load_catalog_builder()
    entries = []
    for task in TASKS:
        path = WEB_ROOT / "data/vision/ultralytics_yolo/yolo26" / f"{task}.yaml"
        safe_regular(path, "Existing task catalog")
        original_bytes = path.read_bytes()
        original = yaml.safe_load(original_bytes)
        record = copy.deepcopy(original)
        need({v.get("size") for v in record["variants"]} == set(SIZES), "Expected all five existing sizes")
        task_added = False
        for variant in record["variants"]:
            size = variant["size"]
            model = (task, size)
            platform_names = [p.get("platform") for p in variant["platforms"]]
            need(len(platform_names) == len(set(platform_names)), f"{task}/{size}: duplicate platform record")
            expected_platforms = {"s600", "s100p", "s100"}
            if "x5" in platform_names:
                expected_platforms.add("x5")
            need(set(platform_names) == expected_platforms,
                 f"{task}/{size}: unexpected existing platform records")
            key = f"ultralytics_yolo/yolo26/{task}/{size}/x5"
            report_name = f"yolo26-{task}-{size}-x5-oe-data.json"
            target_report = WEB_ROOT / "release/reports" / report_name
            has_existing_x5 = "x5" in platform_names
            need((key in source_releases) == has_existing_x5,
                 f"{task}/{size}: X5 catalog platform and release input disagree")
            if model not in selected_set:
                continue
            need(not has_existing_x5 and key not in inputs["releases"] and not target_report.exists(),
                 "X5 entry/report already exists")
            entry, report_bytes = provider(workbench, task, size, release_date)
            need(entry.get("platform") == "x5" and entry.get("status") == "released", "Invalid new X5 record")
            need((entry.get("artifact", {}).get("format"), entry["artifact"].get("march")) == ("bin", "bayes-e"),
                 "X5 catalog artifact must be bayes-e BIN")
            variant["platforms"].append(entry)
            report = json.loads(report_bytes)
            need(report.get("target", {}).get("platform") == "x5" and
                 report.get("provenance", {}).get("artifact_sha256") == entry["artifact"]["sha256"],
                 "OE data does not bind the published X5 artifact")
            inputs["releases"][key] = {"oe_data": f"reports/{report_name}"}
            writes[target_report] = report_bytes
            entries.append({"task": task, "size": size, "platform": "x5",
                            "model_sha256": entry["artifact"]["sha256"]})
            task_added = True
        builder.validate_record(record, path)
        # Existing catalog data, including previously promoted X5 entries,
        # must remain unchanged.
        restored = copy.deepcopy(record)
        for variant in restored["variants"]:
            if (task, variant["size"]) in selected_set:
                need(variant["platforms"][-1].get("platform") == "x5",
                     f"{task}/{variant['size']}: appended X5 record is missing")
                variant["platforms"].pop()
        need(restored == original, "Existing task catalog data changed")
        if task_added:
            writes[path] = yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=100).encode()
            if originals is not None:
                originals[path] = original_bytes
    need(len(entries) == len(selected), "Not every requested model produced a catalog entry")
    need(inputs["models"] == source_inputs["models"] and
         all(inputs["releases"].get(k) == v for k, v in source_releases.items()),
         "Existing releases or cover mappings changed")
    need(len(inputs["releases"]) == len(source_releases) + len(selected),
         "Unexpected catalog entry count")
    writes[WEB_ROOT / "release/inputs.json"] = (json.dumps(inputs, ensure_ascii=False, indent=2) + "\n").encode()
    return writes, {
        "new_entries": entries,
        "new_entry_count": len(entries),
        "existing_entries_preserved": len(source_releases),
        "total_entries": len(inputs["releases"]),
        "files_to_write": len(writes),
        "public_objects_verified": 5 * len(entries),
    }


def commit_incremental_plan(writes: dict[Path, bytes], originals: dict[Path, bytes]) -> None:
    """Replace reviewed catalog files and activate inputs last, with rollback."""
    inputs_path = WEB_ROOT / "release/inputs.json"
    allowed_existing = {inputs_path} | {
        WEB_ROOT / "data/vision/ultralytics_yolo/yolo26" / f"{task}.yaml" for task in TASKS
    }
    targets = list(writes)
    need(targets and targets[-1] == inputs_path, "Release inputs must be activated last")
    need(set(originals) == set(targets) & allowed_existing,
         "Every catalog replacement must bind its reviewed original bytes")

    def check_current(target: Path) -> None:
        need(not target.is_symlink(), f"Unsafe promotion target: {target}")
        if target in originals:
            need(target.is_file() and target.read_bytes() == originals[target],
                 f"Catalog changed after planning: {target}")
        else:
            need(target.parent == WEB_ROOT / "release/reports" and not target.exists(),
                 f"Refusing to replace an existing report: {target}")

    for target in targets:
        check_current(target)
    staged: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target, payload in writes.items():
            fd, name = tempfile.mkstemp(prefix=f".{target.name}.promotion-", dir=target.parent)
            temporary = Path(name)
            staged[target] = temporary
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for target in targets:
            check_current(target)
        for target in targets:
            check_current(target)
            os.replace(staged[target], target)
            committed.append(target)
    except BaseException:
        for target in reversed(committed):
            if target in originals:
                fd, name = tempfile.mkstemp(prefix=f".{target.name}.rollback-", dir=target.parent)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(originals[target])
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, target)
            else:
                target.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench-root", type=Path, default=DEFAULT_WORKBENCH)
    parser.add_argument("--release-date", default=dt.date.today().isoformat())
    parser.add_argument("--models", type=parse_models,
                        help="comma-separated task-size selections, for example cls-n,seg-s; defaults to all 20")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    dt.date.fromisoformat(args.release_date)
    originals = {}
    writes, summary = make_plan(args.workbench_root, args.release_date, models=args.models,
                               originals=originals)
    if args.apply:
        commit_incremental_plan(writes, originals)
    print(json.dumps({"applied": args.apply, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
