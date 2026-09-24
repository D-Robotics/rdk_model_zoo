# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Same-board fixed-image source/unified comparison for the B3 classification batch.

Scope: the four B3 samples (convnext, edgenext, fasternet, fastvit) on X5
8GB/4GB boards, one sample/target/variant per invocation, against the fixed
X5 source pin ``ac115717197920355fc390bb04299b20e6436864``.  On one explicitly
gated board and one prepared artifact this tool runs the fixed source wrapper
(``platforms/x5/samples/vision/<sample>/runtime/python/<sample>.py``) and the
unified entry (``resolve_selection`` -> ``RuntimeModelRunner`` ->
``ClassificationTask``), each with its own complete pre_process -> forward ->
post_process on the same image bytes, resize type, Top-K and scheduling.
Neither side's result substitutes for the other's.

Evidence is written to a brand-new directory: UTC start/end, argv/cwd, git and
code SHA-256 (including the legacy module and its ``utils.py_utils``
dependencies), board identity/version, model/image/label observed SHA-256, SDK
metadata via ``metadata_evidence`` (never ``asdict`` on SDK descriptors), real
exceptions and return codes, every pre input / raw output / result array with
shape/dtype/SHA-256, and the complete comparison.

Source closure: the ``source_ref`` constant alone pins nothing.  Before the
source module executes, every file in its closure — the legacy entry and the
``utils.py_utils`` dependencies it imports — is byte-compared with the pin's
git blob (``git show <pin>:<path>``) against the ``platforms/x5`` snapshot
copies, and the pinned dependency modules are installed into ``sys.modules``
only for the duration of the legacy module execution (restored afterwards, no
persistent pollution).  A closure file that drifted from the pin — the root
``utils/py_utils/file_io.py`` did — is never executed; any mismatch or
unavailable pin object refuses the run.

Decision policy: pre inputs must be non-empty, uint8, finite and exact bytes;
raw outputs must be non-empty, share shape/dtype and be finite with the full
difference reported (raw equality is reported, not asserted); Top-K must have
the requested count, unique IDs, identical ID sequences, scores within
abs 1e-5 and identical label strings (the two label loaders are separate
implementations, so a silent loader change cannot hide); exact ties inside
the top-(k+1) window are recorded with per-ID top-8 evidence for independent
review and never auto-adjudicated into a pass.  This is a fixed-image
migration consistency check, not a dataset accuracy or latency measurement.

Exit codes: 0 all checks passed; 1 comparison completed with failed checks
(arrays retained); 2 execution error (error evidence retained).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    # Direct ``python tools/board_validation/b3_classification_compare.py`` is
    # the supported full-checkout entry; keep the adjustment at this boundary.
    sys.path.insert(0, str(_ROOT))

from samples._shared.assets import (  # noqa: E402
    resolve_asset,
    sha256_file,
    verify_asset_file,
)
from samples._shared.platforms import require_execution_target  # noqa: E402
from samples._shared.runtime_meta import (  # noqa: E402
    RuntimeMetadata,
    metadata_evidence,
)

#: Fixed X5 source commit this comparison is pinned to (rdk_x5 history).
SOURCE_REF = "ac115717197920355fc390bb04299b20e6436864"
SOURCE_GROUP = "x5"

#: Files in the executed fixed-source closure besides the legacy entry: the
#: ``utils.py_utils`` package plus the modules the entries import (the real
#: ``__init__`` also pulls ``visualize``).  Each is byte-compared with the
#: pin before anything is imported (the root ``utils/`` copies have drifted).
_LEGACY_DEP_FILES = (
    "utils/py_utils/__init__.py",
    "utils/py_utils/file_io.py",
    "utils/py_utils/preprocess.py",
    "utils/py_utils/visualize.py",
)
_LEGACY_DEP_MODULES = (
    "utils",
    "utils.py_utils",
    "utils.py_utils.file_io",
    "utils.py_utils.preprocess",
    "utils.py_utils.visualize",
)
_PIN_PATH_BY_MODULE = {
    "utils.py_utils": "utils/py_utils/__init__.py",
    "utils.py_utils.file_io": "utils/py_utils/file_io.py",
    "utils.py_utils.preprocess": "utils/py_utils/preprocess.py",
    "utils.py_utils.visualize": "utils/py_utils/visualize.py",
}

#: B3 acceptance bar: identical Top-K IDs, abs score difference <= 1e-5.
SCORE_ABS_TOLERANCE = 1e-5
#: Per-ID evidence window kept for tie adjudication (B2/B3 evaluator READMEs).
TIE_EVIDENCE_TOPK = 8

#: The B3 batch: legacy class prefix and the sample's canonical test image.
SAMPLES = {
    "convnext": {"prefix": "ConvNeXt", "test_image": "cheetah.JPEG"},
    "edgenext": {"prefix": "EdgeNeXt", "test_image": "Zebra.jpg"},
    "fasternet": {"prefix": "FasterNet", "test_image": "drake.JPEG"},
    "fastvit": {"prefix": "FastViT", "test_image": "bucket.JPEG"},
}

_SHARED_CODE = (
    "assets.py",
    "classification.py",
    "cls_binding.py",
    "image.py",
    "labels.py",
    "model_runner.py",
    "platforms.py",
    "quantization.py",
    "runtime_meta.py",
    "tensor_io.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(path) -> str:
    return sha256_file(Path(path))


def _finite(value) -> Optional[float]:
    """Return a JSON-safe float; non-finite values become None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported evidence value {type(value).__name__}")


def _os_release() -> Optional[Mapping[str, str]]:
    values = {}
    try:
        content = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        if "=" in line:
            key, _, raw = line.partition("=")
            values[key.strip()] = raw.strip().strip('"')
    return {key: values.get(key) for key in ("NAME", "VERSION", "PRETTY_NAME")}


def _meminfo_total_kb() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _board_identity(resolved: str) -> Mapping[str, Any]:
    """Record only public board identity files, leaving absent facts null."""

    from samples._shared.platforms import (
        BOARD_TYPE_PATH,
        DEVICE_TREE_MODEL_PATH,
        SOCINFO_NAME_PATH,
        SOC_NAME_PATH,
    )

    values: dict[str, Any] = {"resolved_target": resolved}
    for path in (SOC_NAME_PATH, SOCINFO_NAME_PATH, BOARD_TYPE_PATH, DEVICE_TREE_MODEL_PATH):
        try:
            values[str(path)] = path.read_text(encoding="utf-8").strip("\x00 \n\r\t")
        except OSError:
            values[str(path)] = None
    values["os_release"] = _os_release()
    values["meminfo_total_kb"] = _meminfo_total_kb()
    return values


def _git_facts() -> Mapping[str, Any]:
    """Snapshot the deployed checkout identity; failures are recorded, not fatal."""

    def git(*args: str):
        try:
            proc = subprocess.run(
                ["git", "-C", str(_ROOT), *args],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if proc.returncode != 0:
            return None, proc.stderr.strip() or f"rc={proc.returncode}"
        return proc.stdout.strip(), None

    head, head_error = git("rev-parse", "HEAD")
    branch, branch_error = git("rev-parse", "--abbrev-ref", "HEAD")
    status, status_error = git("status", "--porcelain")
    files = [line[3:] for line in (status or "").splitlines()] if status is not None else []
    return {
        "head": head,
        "head_error": head_error,
        "branch": branch,
        "branch_error": branch_error,
        "dirty": bool(files),
        "changed_files": files[:200],
        "changed_files_truncated": len(files) > 200,
        "status_error": status_error,
    }


def _host_versions() -> Mapping[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {"python": sys.version, "numpy": np.__version__}
    for name in ("cv2", "scipy"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", None)
    return versions


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))


class _RecordingRuntime:
    """Observe native SDK calls without changing any value passed through."""

    def __init__(self, runtime, record: dict) -> None:
        self._runtime = runtime
        self._record = record

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def run(self, physical):
        name = self.model_names[0]
        flat = physical.get(name, physical) if isinstance(physical, Mapping) else physical
        self._record["inputs"] = {
            str(key): np.array(value, copy=True) for key, value in flat.items()
        }
        result = self._runtime.run(physical)
        out = result.get(name, result) if isinstance(result, Mapping) else result
        self._record["outputs"] = {
            str(key): np.array(value, copy=True) for key, value in out.items()
        }
        return result


def _import_sample_module(sample: str, name: str):
    return importlib.import_module(f"samples.vision.{sample}.runtime.python.{name}")


def _pin_blob_sha256(pin_path: str):
    """Hash one file's content at the fixed pin via the local git object DB.

    Returns ``(sha256_hex, None)`` or ``(None, error)`` when the pin object
    is unavailable (e.g. a stripped deployment); the caller treats that as a
    verification failure, never as an implicit match.
    """

    try:
        proc = subprocess.run(
            ["git", "-C", str(_ROOT), "show", f"{SOURCE_REF}:{pin_path}"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        return None, detail or f"git show rc={proc.returncode}"
    return hashlib.sha256(proc.stdout).hexdigest(), None


def _verify_source_closure(sample: str) -> Mapping[str, Any]:
    """Byte-compare every executed fixed-source closure file against the pin.

    The ``source_ref`` constant alone does not pin anything: the legacy entry
    plus each ``utils.py_utils`` dependency that will actually execute is
    compared with the pin's git blob (against the ``platforms/x5`` snapshot
    copies).  Any mismatch, missing file or unavailable pin object keeps
    ``verified`` false so the caller refuses to run.
    """

    x5_base = _ROOT / "platforms" / SOURCE_GROUP
    targets = [
        (
            "entry",
            f"samples/vision/{sample}/runtime/python/{sample}.py",
            x5_base / "samples" / "vision" / sample / "runtime" / "python" / f"{sample}.py",
        )
    ]
    targets += [
        ("dependency", pin_path, x5_base / pin_path) for pin_path in _LEGACY_DEP_FILES
    ]
    files = []
    for role, pin_path, local_path in targets:
        pin_sha, error = _pin_blob_sha256(pin_path)
        local_sha = _digest(local_path) if local_path.is_file() else None
        files.append(
            {
                "role": role,
                "pin_path": pin_path,
                "local_path": str(local_path),
                "local_sha256": local_sha,
                "pin_sha256": pin_sha,
                "matches_pin": pin_sha is not None
                and local_sha is not None
                and pin_sha == local_sha,
                "error": error,
            }
        )
    return {
        "source_ref": SOURCE_REF,
        "method": (
            f"git show {SOURCE_REF}:<pin_path> byte-compared with the "
            f"{SOURCE_GROUP} snapshot copies before anything is imported"
        ),
        "files": files,
        "verified": all(entry["matches_pin"] for entry in files),
    }


def _load_legacy(sample: str, factory):
    """Execute the fixed X5 source wrapper with an injected recording factory.

    The pin-verified ``utils.py_utils`` closure is installed from the
    ``platforms/x5`` snapshot under the canonical module names for the
    duration of the legacy module execution only: ``sys.modules`` and
    ``sys.path`` are snapshot and restored afterwards, so nothing leaks into
    the process (and the drifted root ``utils/`` copies are never bound).
    ``hbm_runtime`` is bridged the same way (a real SDK stays untouched);
    afterwards the module attribute points at the recording factory.

    Returns ``(module, entry_path, loaded_dependencies)`` where the last
    mapping keeps live references to the pinned dependency modules that the
    entry actually bound.
    """

    x5_base = _ROOT / "platforms" / SOURCE_GROUP
    path = (
        x5_base / "samples" / "vision" / sample / "runtime" / "python" / f"{sample}.py"
    )
    name = f"_b3_compare_legacy_{SOURCE_GROUP}_{sample}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fixed source module: {path}")
    module = importlib.util.module_from_spec(spec)
    bridge = types.ModuleType("hbm_runtime")
    bridge.HB_HBMRuntime = factory
    had_sdk_module = "hbm_runtime" in sys.modules
    path_before = list(sys.path)
    saved_modules = {dep: sys.modules.get(dep) for dep in _LEGACY_DEP_MODULES}

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(x5_base / "utils")]
    py_utils_path = x5_base / "utils" / "py_utils" / "__init__.py"
    py_utils_spec = importlib.util.spec_from_file_location(
        "utils.py_utils",
        py_utils_path,
        submodule_search_locations=[str(x5_base / "utils" / "py_utils")],
    )
    if py_utils_spec is None or py_utils_spec.loader is None:
        raise ImportError(f"Cannot load pinned package init: {py_utils_path}")
    py_utils = importlib.util.module_from_spec(py_utils_spec)
    preprocess_path = x5_base / "utils" / "py_utils" / "preprocess.py"
    preprocess_spec = importlib.util.spec_from_file_location(
        "utils.py_utils.preprocess", preprocess_path
    )
    if preprocess_spec is None or preprocess_spec.loader is None:
        raise ImportError(f"Cannot load pinned dependency: {preprocess_path}")
    preprocess = importlib.util.module_from_spec(preprocess_spec)

    loaded_refs = {}
    try:
        sys.modules["utils"] = utils_pkg
        sys.modules["utils.py_utils"] = py_utils
        # The real package init executes .file_io/.visualize imports, which
        # register those modules in sys.modules from the pinned tree.
        py_utils_spec.loader.exec_module(py_utils)
        if sys.modules.get("utils.py_utils.preprocess") is None:
            sys.modules["utils.py_utils.preprocess"] = preprocess
            preprocess_spec.loader.exec_module(preprocess)
        sys.modules[name] = module
        if not had_sdk_module:
            sys.modules["hbm_runtime"] = bridge
        try:
            spec.loader.exec_module(module)
        finally:
            loaded_refs = {dep: sys.modules.get(dep) for dep in _LEGACY_DEP_MODULES}
            sys.modules.pop(name, None)
            if not had_sdk_module:
                sys.modules.pop("hbm_runtime", None)
    finally:
        for dep, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(dep, None)
            else:
                sys.modules[dep] = previous
        sys.path[:] = path_before
    # The source owns its imported SDK reference; do not mutate a real SDK.
    module.hbm_runtime = bridge
    return module, path, loaded_refs


def _legacy_dependency_resolution(loaded: Mapping[str, Any], closure) -> Mapping[str, Any]:
    """Record the pinned dependency modules the source import actually bound."""

    pin_sha_by_path = {entry["pin_path"]: entry for entry in closure["files"]}
    resolution = {}
    for module_name, pin_path in _PIN_PATH_BY_MODULE.items():
        module = loaded.get(module_name)
        file = getattr(module, "__file__", None)
        entry = {"pin_path": pin_path, "observed_file": str(file) if file else None}
        if file:
            entry["observed_sha256"] = _digest(file)
        closure_entry = pin_sha_by_path.get(pin_path, {})
        entry["pin_sha256"] = closure_entry.get("pin_sha256")
        entry["matches_pin"] = (
            entry.get("observed_sha256") is not None
            and entry["pin_sha256"] is not None
            and entry["observed_sha256"] == entry["pin_sha256"]
        )
        resolution[module_name] = entry
    return resolution


def _tie_groups(ids: np.ndarray, scores: np.ndarray, window: int) -> list[dict]:
    """Groups of exactly equal scores that make Top-K order/membership ambiguous.

    ``window`` is the number of leading ranks considered (Top-K plus the first
    excluded rank).  Exact equality among them means either ordering or
    membership is implementation-defined; ties strictly below the boundary
    rank cannot change the Top-K and are not flagged.
    """

    groups = []
    rank = 0
    while rank < min(window - 1, len(scores) - 1):
        if float(scores[rank]) == float(scores[rank + 1]):
            end = rank + 1
            while end < len(scores) and float(scores[end]) == float(scores[rank]):
                end += 1
            groups.append(
                {
                    "first_rank": rank,
                    "ids": [int(value) for value in ids[rank:end]],
                    "scores": [_finite(value) for value in scores[rank:end]],
                }
            )
            rank = end
        else:
            rank += 1
    return groups


def compare_records(legacy: Mapping[str, Any], unified: Mapping[str, Any], *, top_k: int):
    """Compare both sides' recorded inputs, raw outputs and Top-K results.

    Pure function over the captured records so host tests can pin each rule:
    non-empty uint8 finite inputs with exact bytes (the NV12 ``(1, 3H/2, W, 1)``
    view and the flat buffer may differ in shape, not in bytes); non-empty raw
    outputs sharing shape/dtype, finite, difference reported (not asserted);
    Top-K with the requested count, unique IDs, identical ID sequences, abs
    score difference <= 1e-5 and identical label strings; exact ties inside
    the top-(k+1) window are detected and never pass.  Empty captured dicts
    fail their checks instead of passing vacuously.
    """

    checks: dict[str, bool] = {}
    comparison: dict[str, Any] = {}

    legacy_inputs, unified_inputs = legacy["inputs"], unified["inputs"]
    inputs_present = bool(legacy_inputs) and bool(unified_inputs)
    names_equal = inputs_present and set(legacy_inputs) == set(unified_inputs)
    checks["input_names_equal"] = names_equal
    inputs_report = {}
    bytes_equal_all = names_equal
    uint8_finite_all = names_equal
    for name in sorted(set(legacy_inputs) & set(unified_inputs)):
        left, right = legacy_inputs[name], unified_inputs[name]
        bytes_equal = left.reshape(-1).tobytes() == right.reshape(-1).tobytes()
        uint8_finite = bool(
            left.dtype == np.uint8
            and right.dtype == np.uint8
            and np.isfinite(left).all()
            and np.isfinite(right).all()
        )
        bytes_equal_all = bytes_equal_all and bytes_equal
        uint8_finite_all = uint8_finite_all and uint8_finite
        inputs_report[name] = {
            "legacy_shape": list(left.shape),
            "unified_shape": list(right.shape),
            "legacy_dtype": str(left.dtype),
            "unified_dtype": str(right.dtype),
            "legacy_bytes": len(left.tobytes()),
            "unified_bytes": len(right.tobytes()),
            "uint8_finite_both": uint8_finite,
            "bytes_equal": bytes_equal,
        }
    checks["inputs_exact_bytes"] = bytes_equal_all
    checks["inputs_uint8_finite"] = uint8_finite_all
    comparison["inputs"] = inputs_report

    legacy_raw, unified_raw = legacy["outputs"], unified["outputs"]
    raw_present = bool(legacy_raw) and bool(unified_raw)
    raw_names_equal = raw_present and set(legacy_raw) == set(unified_raw)
    checks["raw_output_names_equal"] = raw_names_equal
    shape_equal_all = raw_names_equal
    dtype_equal_all = raw_names_equal
    finite_all = raw_names_equal
    raw_report = {}
    diffs = {}
    for name in sorted(set(legacy_raw) & set(unified_raw)):
        left, right = legacy_raw[name], unified_raw[name]
        shape_equal = left.shape == right.shape
        dtype_equal = left.dtype == right.dtype
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        shape_equal_all = shape_equal_all and shape_equal
        dtype_equal_all = dtype_equal_all and dtype_equal
        finite_all = finite_all and finite
        entry = {
            "legacy_shape": list(left.shape),
            "unified_shape": list(right.shape),
            "legacy_dtype": str(left.dtype),
            "unified_dtype": str(right.dtype),
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "finite_both": finite,
        }
        if shape_equal and finite:
            diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
            diffs[name] = diff
            entry.update(
                {
                    "max_abs_diff": _finite(np.max(diff)),
                    "mean_abs_diff": _finite(np.mean(diff)),
                    "nonzero_diff_count": int(np.count_nonzero(diff)),
                    "size": int(diff.size),
                    "argmax_diff_flat_index": int(np.argmax(diff)),
                }
            )
        elif shape_equal:
            entry["nonfinite_counts"] = {
                "legacy": int(np.count_nonzero(~np.isfinite(left))),
                "unified": int(np.count_nonzero(~np.isfinite(right))),
            }
        raw_report[name] = entry
    checks["raw_shape_equal"] = shape_equal_all
    checks["raw_dtype_equal"] = dtype_equal_all
    checks["raw_finite"] = finite_all
    comparison["raw_outputs"] = raw_report

    legacy_ids = [int(value) for value in legacy["topk"]["ids"].reshape(-1)]
    unified_ids = [int(value) for value in unified["topk"]["ids"].reshape(-1)]
    legacy_score_values = [
        _finite(value) for value in legacy["topk"]["scores"].reshape(-1)
    ]
    unified_score_values = [
        _finite(value) for value in unified["topk"]["scores"].reshape(-1)
    ]
    legacy_labels = [str(value) for value in legacy["topk"].get("labels", ())]
    unified_labels = [str(value) for value in unified["topk"].get("labels", ())]
    checks["topk_counts_match_request"] = (
        len(legacy_ids) == top_k
        and len(unified_ids) == top_k
        and len(legacy_score_values) == top_k
        and len(unified_score_values) == top_k
    )
    checks["topk_ids_unique"] = (
        len(set(legacy_ids)) == len(legacy_ids)
        and len(set(unified_ids)) == len(unified_ids)
    )
    # The two label loaders are separate implementations (pinned
    # ``load_imagenet_labels`` vs the unified ``load_labels``); identical
    # strings for identical IDs keep a silent loader change from hiding.
    checks["topk_labels_equal"] = (
        len(legacy_labels) == top_k
        and len(unified_labels) == top_k
        and legacy_labels == unified_labels
    )
    ids_equal = legacy_ids == unified_ids
    checks["topk_ids_equal"] = ids_equal
    legacy_scores = {
        int(id_): _finite(score)
        for id_, score in zip(legacy["topk"]["ids"].reshape(-1), legacy["topk"]["scores"].reshape(-1))
    }
    unified_scores = {
        int(id_): _finite(score)
        for id_, score in zip(unified["topk"]["ids"].reshape(-1), unified["topk"]["scores"].reshape(-1))
    }
    legacy_evidence = {
        int(id_): _finite(score)
        for id_, score in zip(legacy["evidence"]["ids"].reshape(-1), legacy["evidence"]["scores"].reshape(-1))
    }
    unified_evidence = {
        int(id_): _finite(score)
        for id_, score in zip(unified["evidence"]["ids"].reshape(-1), unified["evidence"]["scores"].reshape(-1))
    }
    common = sorted(set(legacy_scores) & set(unified_scores))
    pair_diffs = [
        None
        if legacy_scores[id_] is None or unified_scores[id_] is None
        else abs(legacy_scores[id_] - unified_scores[id_])
        for id_ in common
    ]
    diffs_finite = [value for value in pair_diffs if value is not None]
    pairs_complete = len(diffs_finite) == len(common)
    max_common = max(diffs_finite) if diffs_finite else None
    within = (
        ids_equal
        and pairs_complete
        and max_common is not None
        and max_common <= SCORE_ABS_TOLERANCE
    )
    checks["topk_scores_within_tolerance"] = within
    rows = []
    for id_ in sorted(set(legacy_scores) | set(unified_scores)):
        row = {"id": id_}
        for side, topk_map, evidence_map in (
            ("legacy", legacy_scores, legacy_evidence),
            ("unified", unified_scores, unified_evidence),
        ):
            if id_ in topk_map:
                row[f"{side}_score"] = topk_map[id_]
                row[f"{side}_source"] = "topk"
            elif id_ in evidence_map:
                row[f"{side}_score"] = evidence_map[id_]
                row[f"{side}_source"] = "evidence_window_beyond_topk"
            else:
                row[f"{side}_score"] = None
                row[f"{side}_source"] = "absent"
        left, right = row["legacy_score"], row["unified_score"]
        row["abs_diff"] = (
            abs(left - right) if left is not None and right is not None else None
        )
        rows.append(row)
    comparison["topk"] = {
        "legacy_ids": legacy_ids,
        "unified_ids": unified_ids,
        "legacy_labels": legacy_labels,
        "unified_labels": unified_labels,
        "ids_equal": ids_equal,
        "max_common_abs_score_diff": _finite(max_common) if max_common is not None else None,
        "per_id": rows,
    }

    tie_sides = {
        side: _tie_groups(record["evidence"]["ids"], record["evidence"]["scores"], top_k + 1)
        for side, record in (("legacy", legacy), ("unified", unified))
    }
    tie_detected = any(tie_sides.values())
    checks["no_ambiguous_exact_tie"] = not tie_detected
    comparison["tie"] = {
        "detected": tie_detected,
        "window_ranks": top_k + 1,
        "sides": tie_sides,
        "policy": (
            "Exact score ties inside the top-(k+1) window make Top-K ordering or "
            "membership implementation-defined; the run is not recorded as passed. "
            "Per-ID top-8 evidence from both sides is retained for independent "
            "adjudication; this tool never auto-relaxes a tie into a pass."
        ),
    }
    comparison["top8_per_id"] = {
        side: [
            {"rank": rank, "id": int(id_), "score": _finite(score)}
            for rank, (id_, score) in enumerate(
                zip(record["evidence"]["ids"].reshape(-1), record["evidence"]["scores"].reshape(-1))
            )
        ]
        for side, record in (("legacy", legacy), ("unified", unified))
    }
    comparison["checks"] = checks
    comparison["tolerances"] = {
        "inputs": "non-empty uint8 finite; exact bytes across NV12 view shapes",
        "raw_outputs": "non-empty; shape/dtype/finite gated; difference reported, equality not asserted",
        "topk_score_abs": SCORE_ABS_TOLERANCE,
        "topk_labels": "identical strings for identical IDs (both loaders)",
        "tie": "exact equality inside top-(k+1) window",
    }
    comparison["decision_policy"] = (
        "Inputs must be non-empty uint8 finite and byte-identical; raw outputs "
        "non-empty with equal shape/dtype and finite, full difference reported; "
        "Top-K must have the requested count, unique IDs, identical ID sequences "
        "and label strings, per-ID score abs diff <= 1e-5; exact ties inside the "
        "top-(k+1) window are recorded with per-ID top-8 evidence and never "
        "auto-adjudicated or relaxed into a pass."
    )
    comparison["passed"] = all(checks.values())
    comparison["raw_diffs"] = diffs
    return comparison


def _save_arrays(directory: Path, records: Mapping[str, Any], comparison) -> Mapping[str, Any]:
    """Persist every captured array as .npy with shape/dtype/SHA-256 evidence."""

    saved = {}

    def save(filename: str, array: np.ndarray) -> None:
        if Path(filename).name != filename:
            raise ValueError(f"Unsafe tensor evidence filename {filename!r}.")
        path = directory / filename
        np.save(path, array, allow_pickle=False)
        saved[filename] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _digest(path),
        }

    for side, record in records.items():
        for kind, label in (("inputs", "input"), ("outputs", "raw")):
            for name, value in record.get(kind, {}).items():
                save(f"{side}_{label}_{_safe_name(name)}.npy", value)
        for section in ("topk", "evidence"):
            captured = record.get(section)
            if not captured:
                continue
            save(f"{side}_{section}_ids.npy", captured["ids"])
            save(f"{side}_{section}_scores.npy", captured["scores"])
    for name, diff in comparison.get("raw_diffs", {}).items():
        save(f"raw_abs_diff_{_safe_name(name)}.npy", diff)
    return saved


def _validate_scheduling(priority: int, bpu_cores: Optional[Sequence[int]]) -> list[int]:
    if type(priority) is not int or not 0 <= priority <= 255:
        raise ValueError("priority must be an integer in 0..255.")
    if bpu_cores is None:
        return [0]
    cores = list(bpu_cores)
    if not cores or any(type(core) is not int or core < 0 for core in cores):
        raise ValueError("bpu-cores must be a nonempty list of nonnegative indexes.")
    return cores


def run_comparison(
    selection,
    image_path,
    image,
    label_path,
    output_dir,
    *,
    top_k: int = 5,
    resize_type: int = 1,
    priority: int = 0,
    bpu_cores=None,
    runtime_factory=None,
):
    """Capture source baseline then unified inference on the same exact files.

    ``runtime_factory`` is an offline test seam; production invocation still
    always checks exact hardware identity.  A new evidence directory is
    mandatory.  Failed execution writes error evidence then re-raises; a failed
    comparison returns ``passed=False`` with every array intact.  The CLI maps
    execution failure to status 2 and comparison failure to status 1.
    """

    sample = selection.sample_id
    if sample not in SAMPLES:
        raise ValueError(f"Sample {sample!r} is outside the B3 classification batch.")
    if not 1 <= top_k <= selection.contract.class_count:
        raise ValueError(f"top_k must be in 1..{selection.contract.class_count}.")
    if resize_type not in (0, 1):
        raise ValueError("resize_type must be 0 (direct) or 1 (letterbox).")
    cores = _validate_scheduling(priority, bpu_cores)
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(f"Evidence directory must not already exist: {directory}.")
    actual_target = require_execution_target(selection.target)
    if actual_target != selection.target:
        raise ValueError(f"Target mismatch: requested {selection.target}, detected {actual_target}.")

    evidence_topk = min(max(top_k + 1, TIE_EVIDENCE_TOPK), selection.contract.class_count)
    asset = resolve_asset(selection.asset_id)
    records = {side: {"inputs": {}, "outputs": {}} for side in ("legacy", "unified")}
    summary = {
        "tool": "tools/board_validation/b3_classification_compare.py",
        "schema_version": 1,
        "measurement": (
            "fixed-image source/unified migration consistency on one board; "
            "not dataset accuracy or latency"
        ),
        "sample": sample,
        "target": selection.target,
        "variant": selection.variant,
        "asset_id": selection.asset_id,
        "source_ref": SOURCE_REF,
        "source_entry": str(
            _ROOT / "platforms" / SOURCE_GROUP / "samples" / "vision" / sample
            / "runtime" / "python" / f"{sample}.py"
        ),
        "unified_entry": (
            f"samples/vision/{sample}/runtime/python (resolve_selection -> "
            "RuntimeModelRunner -> ClassificationTask)"
        ),
        "started_utc": _now(),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "output_dir": str(directory),
        "board": _board_identity(actual_target),
        "git": _git_facts(),
        "host_versions": _host_versions(),
        "sdk": None,
        "artifacts": {
            "model": {
                "path": str(Path(selection.model_path).resolve()),
                "publisher_sha256": asset.sha256,
                "observed_sha256": None,
                "matches_publisher": None,
            }
        },
        "image": {"path": str(Path(image_path).resolve()), "sha256": None,
                  "decoded_shape": list(image.shape), "decoded_dtype": str(image.dtype)},
        "labels": {"path": str(Path(label_path).resolve()), "sha256": None},
        "parameters": {
            "top_k": top_k,
            "resize_type": resize_type,
            "evidence_topk": evidence_topk,
            "priority": priority,
            "bpu_cores": cores,
        },
        "code_sha256": {},
        "source_closure": None,
        "legacy_dependency_resolution": {},
        "metadata": {side: None for side in records},
        "records": None,
        "comparison": None,
        "passed": False,
        "return_code": None,
        "error": None,
        "finished_utc": None,
    }
    directory.mkdir(parents=True, exist_ok=False)
    failure = None
    try:
        observed = verify_asset_file(asset, selection.model_path)
        summary["artifacts"]["model"]["observed_sha256"] = observed
        summary["artifacts"]["model"]["matches_publisher"] = (
            None if asset.sha256 is None else observed == asset.sha256.lower()
        )
        summary["image"]["sha256"] = _digest(image_path)
        summary["labels"]["sha256"] = _digest(label_path)

        if runtime_factory is None:
            from samples._shared.model_runner import _default_runtime_factory

            runtime_factory = _default_runtime_factory()
            sdk = sys.modules.get("hbm_runtime")
            summary["sdk"] = {
                "module_file": getattr(sdk, "__file__", None),
                "version": getattr(sdk, "__version__", None),
                "injected": False,
            }
        else:
            summary["sdk"] = {"module_file": None, "version": None, "injected": True}

        def recording_factory(side):
            def create(path):
                runtime = runtime_factory(str(path))
                # Projected without copying SDK quant descriptors (asdict
                # deepcopies and the board QuantParams refuses it).
                summary["metadata"][side] = metadata_evidence(
                    RuntimeMetadata.from_runtime(runtime)
                )
                return _RecordingRuntime(runtime, records[side])

            return create

        # The source_ref constant alone does not pin anything: refuse to run
        # unless every closure file actually matches the pin's git blob.
        closure = _verify_source_closure(sample)
        summary["source_closure"] = closure
        if not closure["verified"]:
            broken = "; ".join(
                f"{entry['pin_path']} ({entry['error'] or 'hash mismatch'})"
                for entry in closure["files"]
                if not entry["matches_pin"]
            )
            raise ValueError(
                "Fixed-source closure does not match pin "
                f"{SOURCE_REF}: {broken}. The source entry and its "
                "utils.py_utils dependencies must be byte-identical with the "
                "pin before any comparison runs."
            )

        prefix = SAMPLES[sample]["prefix"]
        legacy_module, legacy_path, loaded_deps = _load_legacy(
            sample, recording_factory("legacy")
        )
        summary["legacy_dependency_resolution"] = _legacy_dependency_resolution(
            loaded_deps, closure
        )

        legacy_config = getattr(legacy_module, f"{prefix}Config")(
            model_path=str(selection.model_path),
            label_file=str(label_path),
            resize_type=resize_type,
            topk=top_k,
        )
        legacy_model = getattr(legacy_module, prefix)(legacy_config)
        legacy_model.set_scheduling_params(priority=priority, bpu_cores=cores)
        legacy_prepared = legacy_model.pre_process(image)
        legacy_raw = legacy_model.forward(legacy_prepared)
        legacy_topk = legacy_model.post_process(legacy_raw)
        legacy_evidence = legacy_model.post_process(legacy_raw, topk=evidence_topk)
        records["legacy"]["topk"] = {
            "ids": np.asarray(legacy_topk[0], dtype=np.int64),
            "scores": np.asarray(legacy_topk[1]),
            "labels": [str(value) for value in legacy_topk[2]],
        }
        records["legacy"]["evidence"] = {
            "ids": np.asarray(legacy_evidence[0], dtype=np.int64),
            "scores": np.asarray(legacy_evidence[1]),
        }

        runner_mod = _import_sample_module(sample, "model_runner")
        classification_mod = _import_sample_module(sample, "classification")
        labels_mod = _import_sample_module(sample, "labels")
        labels = labels_mod.load_labels(Path(label_path))
        runner = runner_mod.RuntimeModelRunner(
            selection, runtime_factory=recording_factory("unified")
        )
        binding = runner.load()
        runner.set_scheduling_params(priority=priority, bpu_cores=cores)
        task = classification_mod.ClassificationTask(
            runner,
            binding,
            top_k=top_k,
            labels=labels,
            resize_type=resize_type,
        )
        unified_prepared = task.pre_process(image)
        unified_raw = task.forward(unified_prepared)
        unified_topk = task.post_process(unified_raw)
        evidence_task = classification_mod.ClassificationTask(
            runner,
            binding,
            top_k=evidence_topk,
            labels=labels,
            resize_type=resize_type,
        )
        unified_evidence = evidence_task.post_process(unified_raw)
        records["unified"]["topk"] = {
            "ids": np.asarray(unified_topk.class_ids, dtype=np.int64),
            "scores": np.asarray(unified_topk.scores),
            "labels": [str(value) for value in unified_topk.labels],
        }
        records["unified"]["evidence"] = {
            "ids": np.asarray(unified_evidence.class_ids, dtype=np.int64),
            "scores": np.asarray(unified_evidence.scores),
        }

        comparison = compare_records(records["legacy"], records["unified"], top_k=top_k)
        raw_diffs = comparison.pop("raw_diffs")
        summary["comparison"] = comparison
        summary["passed"] = comparison["passed"]

        code_paths = [Path(__file__).resolve(), legacy_path]
        code_paths += [
            Path(file)
            for name in ("utils.py_utils.file_io", "utils.py_utils.preprocess")
            if (file := getattr(loaded_deps.get(name), "__file__", None))
        ]
        code_paths += [_ROOT / "samples" / "_shared" / name for name in _SHARED_CODE]
        code_paths += sorted(
            (_ROOT / "samples" / "vision" / sample / "runtime" / "python").glob("*.py")
        )
        summary["code_sha256"] = {
            str(path.relative_to(_ROOT)) if path.is_relative_to(_ROOT) else str(path): _digest(path)
            for path in code_paths
        }
        summary["records"] = {
            side: {
                "inputs": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in record["inputs"].items()
                },
                "raw_outputs": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in record["outputs"].items()
                },
                "topk": record["topk"],
                "evidence": record["evidence"],
            }
            for side, record in records.items()
        }
        summary["arrays"] = _save_arrays(directory, records, {"raw_diffs": raw_diffs})
    except Exception as exc:  # noqa: BLE001 - recorded verbatim, then re-raised
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        failure = exc
    finally:
        if summary.get("arrays") is None:
            summary["arrays"] = _save_arrays(directory, records, {"raw_diffs": {}})
        summary["return_code"] = 2 if failure is not None else (0 if summary["passed"] else 1)
        summary["finished_utc"] = _now()
        (directory / "comparison.json").write_text(
            json.dumps(summary, indent=2, default=_json_default, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if failure is not None:
        raise failure
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "B3 classification batch: same-board fixed-image source/unified "
            "comparison for one sample/target/variant; no downloads."
        )
    )
    parser.add_argument("--sample", required=True, choices=sorted(SAMPLES))
    parser.add_argument("--target", required=True, choices=("x5", "s100", "s100p", "s600"))
    parser.add_argument(
        "--variant",
        default=None,
        help="Exact published variant; default follows the sample's own selection semantics.",
    )
    parser.add_argument(
        "--asset-id", default=None,
        help="Exact qualified manifest reference group:sample:filename.",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Existing artifact path; requires --asset-id (exact reference), never downloads.",
    )
    parser.add_argument(
        "--test-img", default=None,
        help="BGR input image (default: the sample's bundled test image, e.g. "
        "convnext -> cheetah.JPEG, edgenext -> Zebra.jpg, fasternet -> drake.JPEG, "
        "fastvit -> bucket.JPEG).",
    )
    parser.add_argument(
        "--label-file", default=None,
        help="ImageNet label file (default: datasets/imagenet/imagenet_classes.names).",
    )
    parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    parser.add_argument("--resize-type", dest="resize_type", type=int, choices=(0, 1), default=1)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="New evidence directory; existing paths are rejected.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        binding_mod = _import_sample_module(args.sample, "model_binding")
        selection = binding_mod.resolve_selection(
            args.target,
            asset_id=args.asset_id,
            variant=args.variant,
            model_path=args.model_path,
        )
        image_path = (
            Path(args.test_img).expanduser()
            if args.test_img is not None
            else _ROOT / "samples" / "vision" / args.sample / "test_data"
            / SAMPLES[args.sample]["test_image"]
        )
        label_path = (
            Path(args.label_file).expanduser()
            if args.label_file is not None
            else _ROOT / "datasets" / "imagenet" / "imagenet_classes.names"
        )
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"test image not found or unreadable: {image_path}")
        summary = run_comparison(
            selection,
            image_path,
            image,
            label_path,
            args.output_dir,
            top_k=args.top_k,
            resize_type=args.resize_type,
            priority=args.priority,
            bpu_cores=args.bpu_cores,
        )
        comparison = summary["comparison"]

        def side_topk(side):
            captured = summary["records"][side]["topk"]
            return {
                "ids": [int(value) for value in captured["ids"].reshape(-1)],
                "scores": [_finite(value) for value in captured["scores"].reshape(-1)],
                "labels": captured["labels"],
            }

        print(
            json.dumps(
                {
                    "sample": summary["sample"],
                    "variant": summary["variant"],
                    "target": summary["target"],
                    "asset_id": summary["asset_id"],
                    "passed": summary["passed"],
                    "return_code": summary["return_code"],
                    "checks": comparison["checks"],
                    "tie_detected": comparison["tie"]["detected"],
                    "legacy_topk": side_topk("legacy"),
                    "unified_topk": side_topk("unified"),
                    "evidence": str(Path(summary["output_dir"]) / "comparison.json"),
                },
                default=_json_default,
            )
        )
        return summary["return_code"]
    except Exception as exc:  # noqa: BLE001 - the CLI maps any failure to status 2
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
