English | [简体中文](./README_cn.md)

# R3D-18 Evaluation Record

<a id="dataset"></a>
## Dataset

The supplied functional input is the preprocessed `test_data/video0.npy` clip, shape `(1,3,16,112,112)`, dtype float32. It represents a 16-frame archery sample. `test_data/kinetics_classnames.json` contains the 400-entry Kinetics name-to-id mapping used by the CLI; names are decoded into id-to-name labels with the source removal of literal double quotes.

This directory does not contain the full Kinetics-400 dataset, a video decoder, frame extraction code, or a dataset download command. The clip's original acquisition and preprocessing command are not recorded.

<a id="environment"></a>
## Environment

- Unified host check: repository `.venv`, Python 3.14.7, NumPy, and PyYAML; no HBM or board is needed for the fixture tests.
- Board functional check: RDK S100 with the matching `hbm_runtime`, prepared `model/s100/r3d_18.hbm`, and recognized S100 identity.
- The source performance table was produced with `hrt_model_exec`; its exact invocation, image, runtime version, and raw output files are not present, so it is a historical source record rather than a reproduced benchmark.

<a id="command"></a>
## Evaluation Commands

Unified host contract check:

```bash
# cwd: repository root
.venv/bin/python -m unittest discover -s samples/vision/3dresnet/tests -v
# expect: all discovered tests OK; this checks source-compatible host behavior only
```

Unified S100 functional smoke command:

```bash
# cwd: repository root; prerequisites: explicit model download and S100 board
bash samples/vision/3dresnet/model/download.sh s100
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm \
  --model-path samples/vision/3dresnet/model/s100/r3d_18.hbm \
  --test-clip samples/vision/3dresnet/test_data/video0.npy \
  --label-file samples/vision/3dresnet/test_data/kinetics_classnames.json \
  --top-k 5 --priority 0 --bpu-cores 0
# expect: exit code 0 and JSON with five predictions; this command is not-run in this migration
```

The old source says it used `hrt_model_exec` for performance, but does not preserve a complete copyable command or its input/output file arguments. No replacement command is invented here.

For a future migration check, the following self-contained legacy/unified baseline uses the same S100 model and `video0.npy`. It performs the concrete S100 identity check before importing the legacy wrapper or loading the SDK, saves both sides' prepared input and raw output in one unique UTC directory, and compares metadata, raw values, and source Top-K results. It is a host-authored recipe/fixture only: **not-run** in this migration and not a board result.

```bash
# cwd: repository root; prerequisites: prepared HBM, source tree, and recognized S100 board
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="/tmp/3dresnet-baseline/${RUN_ID}"
mkdir -p "${OUT_DIR}"
OUT_DIR="${OUT_DIR}" python3 - <<'PY'
import importlib
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
    raise AssertionError("Migration comparison failed; inspect full raw arrays and comparison.json. No automatic tie exemption.")
PY
# expect: source_input.npy, unified_input.npy, source_raw.npy, unified_raw.npy,
# comparison.json, and stdout showing the exact shape/dtype and tolerances.
# This is a copyable host-authored recipe/fixture; it is not a current board result.
```

The recipe requires exact input and output shape/dtype agreement, raw `allclose(atol=1e-5, rtol=0)`, and source Top-K score agreement within `atol=1e-6, rtol=0`. Top-K IDs must match. Any ID mismatch fails and preserves comparison.json for independent review, including exact ties; no approximate-tie or boundary tolerance automatically waives the result. These are numerical comparisons, never a claim of bitwise equality of board raw outputs. The source `archery` display remains a reference; this recipe does not measure it.

<a id="metrics"></a>
## Metrics

- **Top-1:** the class ID at rank one after numerically stable softmax and descending probability order.
- **Top-K:** the first `K` class IDs and float32 probabilities, with `K` controlled by `--top-k` (default 5).
- **Functional label check:** the source reference expects `archery` at Top-1 for `video0.npy`; this is not a current board result.
- **Performance:** the table below preserves the source thread-performance record. “Total Latency” and “Average Latency” are milliseconds; FPS is throughput. The values are not measured by this migration and must not be read as current S100 evidence.

| Threads | Frames | Total Latency (ms) | Average Latency (ms) | FPS |
| --- | --- | --- | --- | --- |
| 1 | 100 | 18267.76 | 182.68 | 5.47 |
| 2 | 100 | 18291.76 | 182.93 | 10.82 |
| 4 | 100 | 18501.06 | 185.03 | 21.07 |
| 8 | 100 | 24743.56 | 249.19 | 30.74 |

The source additionally records approximate BPU occupancy 5.2%, ION memory 91.9 MB, read bandwidth 533, and write bandwidth 304. Units and measurement setup are not fully recorded; these are retained as source notes only.

<a id="outputs"></a>
## Outputs

The unified functional command writes its JSON report to stdout and does not create a result file. Each prediction contains `class_id`, `score`, and `label`; raw model output is not saved by the CLI. Host test output is the unittest log. The preserved source screenshots show the source archery frame and Top-5 display:

![Archery frame](../test_data/readme_img/image-4.png)
![Top-5 result](../test_data/readme_img/image-5.png)

<a id="reference-results"></a>
## Reference Results

| Reference | Status | Provenance |
| --- | --- | --- |
| `video0.npy` Top-1 `archery` | source functional reference; board not-run | source evaluator README and screenshot |
| Four-row thread-performance table | source historical record; not reproduced | source evaluator README |
| BPU/ION/bandwidth notes | source historical record; not reproduced | source evaluator README and screenshot |

No bitwise raw-output comparison, accuracy claim, or current FPS claim is made.

<a id="boundaries"></a>
## Boundaries

- There is no full-dataset evaluator implementation in this sample.
- There is no current board execution record, raw output archive, or UTC result directory from this migration.
- There is no complete source `hrt_model_exec` command, so the legacy performance record cannot be reproduced from repository contents alone.
- Host tests validate preprocessing, finite/shape/dtype guards, dynamic tensor names, source softmax/Top-K parity, labels, CLI gates, and mocked download delegation; they do not validate HBM execution.

The source additional-metrics screenshot is retained for traceability:

![Additional metrics](../test_data/readme_img/image-6.png)
