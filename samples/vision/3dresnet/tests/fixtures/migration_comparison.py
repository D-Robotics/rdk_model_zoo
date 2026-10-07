# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test-only 3DResNet source-parity recipe (no code executes at import).

The constants below hold the exact ``python3 - <<'PY'`` heredocs formerly
published in ``samples/vision/3dresnet/evaluator/README.md`` and
``README_cn.md`` at commit f773f3542f01. The customer-facing evaluator
documents no longer carry the migration recipe; it is preserved here so the
host suite still executes the identical source-vs-unified comparison under
fake-SDK fixtures. The English and Chinese heredocs differ only in comments;
both variants are kept explicitly rather than discarding one. Only
``test_readme_contract.py`` compiles and executes these strings.
"""

README_MD = r'''import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np

repo = Path.cwd()
out = Path(os.environ["OUT_DIR"])
target = "s100"
asset_id = "s:3dresnet:s100/r3d_18.hbm"
model_path = repo / "samples/vision/3dresnet/model/s100/r3d_18.hbm"
clip_path = repo / "samples/vision/3dresnet/test_data/video0.npy"

# This is the only board gate. It must precede both the legacy import and SDK use.
platforms = importlib.import_module("samples._shared.platforms")
platforms.require_execution_target(target)
if not model_path.is_file():
    raise FileNotFoundError(model_path)

# Import the immutable source wrapper only after identity is established. Its
# source helper expects platforms/s on sys.path and imports hbm_runtime.
sys.path.insert(0, str(repo / "platforms/s"))
legacy_spec = importlib.util.spec_from_file_location(
    "source_r3d18_resnet3d",
    repo / "platforms/s/samples/vision/3dresnet/runtime/python/resnet3d.py",
)
if legacy_spec is None or legacy_spec.loader is None:
    raise RuntimeError("cannot load source R3D-18 wrapper")
legacy_mod = importlib.util.module_from_spec(legacy_spec)
sys.modules[legacy_spec.name] = legacy_mod
legacy_spec.loader.exec_module(legacy_mod)

# Unified modules are loaded by their repository package names; no runtime-path
# insertion or bare module import is used for the migrated implementation.
binding = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
runner_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_runner")
task_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.classification")
labels_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.labels")
selection = binding.resolve_selection(
    target, asset_id=asset_id, model_path=model_path)

# Both paths use the same model and the same source-preprocessed fixture.
legacy = legacy_mod.ResNet3D(legacy_mod.ResNet3DConfig(str(model_path)))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
runner = runner_mod.RuntimeModelRunner(selection)
bound = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
clip = np.load(clip_path, allow_pickle=False)
task = task_mod.VideoClassificationTask(
    runner, bound, top_k=5,
    labels=labels_mod.load_labels(repo / "samples/vision/3dresnet/test_data/kinetics_classnames.json"))
source_input = legacy.pre_process(clip)[legacy.model_name][legacy.input_name]
unified_prepared = task.pre_process(clip)
unified_input = unified_prepared.tensors[bound.input_name]
if source_input.shape != unified_input.shape or source_input.dtype != unified_input.dtype:
    raise AssertionError("legacy/unified input shape or dtype differs")
if not np.array_equal(source_input, unified_input):
    raise AssertionError("legacy/unified prepared input differs")
np.save(out / "source_input.npy", source_input, allow_pickle=False)
np.save(out / "unified_input.npy", unified_input, allow_pickle=False)

source_raw_nested = legacy.forward({legacy.model_name: {legacy.input_name: source_input}})
source_raw = np.asarray(source_raw_nested[legacy.model_name][legacy.output_name])
unified_raw_map = runner(unified_prepared.tensors)
unified_raw = np.asarray(unified_raw_map[bound.output_name])
if source_raw.shape != unified_raw.shape or source_raw.dtype != unified_raw.dtype:
    raise AssertionError("legacy/unified raw shape or dtype differs")
if not np.isfinite(source_raw).all() or not np.isfinite(unified_raw).all():
    raise AssertionError("legacy/unified raw output is not finite")
np.save(out / "source_raw.npy", source_raw, allow_pickle=False)
np.save(out / "unified_raw.npy", unified_raw, allow_pickle=False)
raw_close = bool(np.allclose(source_raw, unified_raw, rtol=0.0, atol=1e-5))

source_topk = legacy.post_process(source_raw_nested, top_k=5)
unified_result = task.post_process(unified_raw_map)
source_ids = [int(item[0]) for item in source_topk]
source_scores = [float(item[1]) for item in source_topk]
unified_ids = [int(item) for item in unified_result.class_ids]
unified_scores = [float(item) for item in unified_result.scores]
ids_equal = source_ids == unified_ids
scores_equal = bool(np.allclose(source_scores, unified_scores, rtol=0.0, atol=1e-6))
passed = raw_close and ids_equal and scores_equal

metadata = {
    "target": target, "asset_id": asset_id, "model_path": str(model_path),
    "source_input": {"shape": list(source_input.shape), "dtype": str(source_input.dtype)},
    "unified_input": {"shape": list(unified_input.shape), "dtype": str(unified_input.dtype)},
    "source_raw": {"name": legacy.output_name, "shape": list(source_raw.shape), "dtype": str(source_raw.dtype)},
    "unified_raw": {"name": bound.output_name, "shape": list(unified_raw.shape), "dtype": str(unified_raw.dtype)},
    "raw_allclose": raw_close, "raw_atol": 1e-5, "raw_rtol": 0.0,
    "source_topk_ids": source_ids, "unified_topk_ids": unified_ids,
    "source_topk_scores": source_scores, "unified_topk_scores": unified_scores,
    "score_atol": 1e-6, "score_rtol": 0.0,
    "ids_equal": ids_equal, "scores_equal": scores_equal,
    "passed": passed, "id_mismatch_requires_review": not ids_equal,
}
(out / "comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"out_dir": str(out), **metadata}, indent=2))
if not passed:
    raise AssertionError("Migration comparison failed; inspect full raw arrays and comparison.json. No automatic tie exemption.")'''

README_CN_MD = r'''import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import numpy as np

repo = Path.cwd()
out = Path(os.environ["OUT_DIR"])
target = "s100"
asset_id = "s:3dresnet:s100/r3d_18.hbm"
model_path = repo / "samples/vision/3dresnet/model/s100/r3d_18.hbm"
clip_path = repo / "samples/vision/3dresnet/test_data/video0.npy"

# 这是唯一的板卡门禁，必须先于 source 导入和 SDK 使用。
platforms = importlib.import_module("samples._shared.platforms")
platforms.require_execution_target(target)
if not model_path.is_file():
    raise FileNotFoundError(model_path)

# 身份确认后才导入未修改的 source wrapper；它要求 platforms/s 在 sys.path
# 中并会导入 hbm_runtime。
sys.path.insert(0, str(repo / "platforms/s"))
legacy_spec = importlib.util.spec_from_file_location(
    "source_r3d18_resnet3d",
    repo / "platforms/s/samples/vision/3dresnet/runtime/python/resnet3d.py",
)
if legacy_spec is None or legacy_spec.loader is None:
    raise RuntimeError("cannot load source R3D-18 wrapper")
legacy_mod = importlib.util.module_from_spec(legacy_spec)
sys.modules[legacy_spec.name] = legacy_mod
legacy_spec.loader.exec_module(legacy_mod)

# 统一实现使用完整 repository package 名称导入；不插入 runtime 目录，也不
# 使用裸模块名。
binding = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
runner_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_runner")
task_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.classification")
labels_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.labels")
selection = binding.resolve_selection(
    target, asset_id=asset_id, model_path=model_path)

# 两条路径使用同一模型和同一个 source-preprocessed fixture。
legacy = legacy_mod.ResNet3D(legacy_mod.ResNet3DConfig(str(model_path)))
legacy.set_scheduling_params(priority=0, bpu_cores=[0])
runner = runner_mod.RuntimeModelRunner(selection)
bound = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
clip = np.load(clip_path, allow_pickle=False)
task = task_mod.VideoClassificationTask(
    runner, bound, top_k=5,
    labels=labels_mod.load_labels(repo / "samples/vision/3dresnet/test_data/kinetics_classnames.json"))
source_input = legacy.pre_process(clip)[legacy.model_name][legacy.input_name]
unified_prepared = task.pre_process(clip)
unified_input = unified_prepared.tensors[bound.input_name]
if source_input.shape != unified_input.shape or source_input.dtype != unified_input.dtype:
    raise AssertionError("legacy/unified input shape or dtype differs")
if not np.array_equal(source_input, unified_input):
    raise AssertionError("legacy/unified prepared input differs")
np.save(out / "source_input.npy", source_input, allow_pickle=False)
np.save(out / "unified_input.npy", unified_input, allow_pickle=False)

source_raw_nested = legacy.forward({legacy.model_name: {legacy.input_name: source_input}})
source_raw = np.asarray(source_raw_nested[legacy.model_name][legacy.output_name])
unified_raw_map = runner(unified_prepared.tensors)
unified_raw = np.asarray(unified_raw_map[bound.output_name])
if source_raw.shape != unified_raw.shape or source_raw.dtype != unified_raw.dtype:
    raise AssertionError("legacy/unified raw shape or dtype differs")
if not np.isfinite(source_raw).all() or not np.isfinite(unified_raw).all():
    raise AssertionError("legacy/unified raw output is not finite")
np.save(out / "source_raw.npy", source_raw, allow_pickle=False)
np.save(out / "unified_raw.npy", unified_raw, allow_pickle=False)
raw_close = bool(np.allclose(source_raw, unified_raw, rtol=0.0, atol=1e-5))

source_topk = legacy.post_process(source_raw_nested, top_k=5)
unified_result = task.post_process(unified_raw_map)
source_ids = [int(item[0]) for item in source_topk]
source_scores = [float(item[1]) for item in source_topk]
unified_ids = [int(item) for item in unified_result.class_ids]
unified_scores = [float(item) for item in unified_result.scores]
ids_equal = source_ids == unified_ids
scores_equal = bool(np.allclose(source_scores, unified_scores, rtol=0.0, atol=1e-6))
passed = raw_close and ids_equal and scores_equal

metadata = {
    "target": target, "asset_id": asset_id, "model_path": str(model_path),
    "source_input": {"shape": list(source_input.shape), "dtype": str(source_input.dtype)},
    "unified_input": {"shape": list(unified_input.shape), "dtype": str(unified_input.dtype)},
    "source_raw": {"name": legacy.output_name, "shape": list(source_raw.shape), "dtype": str(source_raw.dtype)},
    "unified_raw": {"name": bound.output_name, "shape": list(unified_raw.shape), "dtype": str(unified_raw.dtype)},
    "raw_allclose": raw_close, "raw_atol": 1e-5, "raw_rtol": 0.0,
    "source_topk_ids": source_ids, "unified_topk_ids": unified_ids,
    "source_topk_scores": source_scores, "unified_topk_scores": unified_scores,
    "score_atol": 1e-6, "score_rtol": 0.0,
    "ids_equal": ids_equal, "scores_equal": scores_equal,
    "passed": passed, "id_mismatch_requires_review": not ids_equal,
}
(out / "comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps({"out_dir": str(out), **metadata}, indent=2))
if not passed:
    raise AssertionError("Migration comparison failed; inspect full raw arrays and comparison.json. No automatic tie exemption.")'''
