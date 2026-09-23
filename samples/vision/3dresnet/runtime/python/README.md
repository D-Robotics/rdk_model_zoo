English | [简体中文](./README_cn.md)

# Python Runtime — R3D-18

<a id="environment"></a>
## Environment

- Board execution: RDK S100 with the matching `hbm_runtime` Python environment; exact board image/runtime versions were not recorded and board execution is not-run.
- Host verification: repository `.venv` with Python 3.14.7, `numpy`, and `PyYAML` (`requirements-host.txt` lists the host dependencies).
- The runtime accepts a prepared NumPy clip. It does not import a video decoder, read video frames, resize images, or normalize pixels.
- SDK-free operations are `--help`, `--list-models`, and explicit `--dry-run`; they do not construct `hbm_runtime`.

<a id="usage"></a>
## Usage

Prepare the model first. From the repository root, the explicit board command is:

```bash
# cwd: repository root; prerequisite: model/download.sh has placed the HBM
python3 samples/vision/3dresnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:3dresnet:s100/r3d_18.hbm
# expect: exit code 0 and JSON predictions; board identity must resolve to S100
```

The same command from the runtime directory is:

```bash
# cwd: samples/vision/3dresnet/runtime/python
bash run.sh --target s100 --asset-id s:3dresnet:s100/r3d_18.hbm
```

`run.sh` is only a CLI delegate. It does not download the model. The host fixture pipeline is verified with:

```bash
# cwd: repository root
.venv/bin/python -m unittest discover -s samples/vision/3dresnet/tests -v
# expect: all discovered tests OK; this does not prove board execution
```

<a id="parameters"></a>
## Parameters

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | `auto`/`s100` | `auto` | Board target; `auto` detects identity only for preparation/execution resolution |
| `--asset-id` | string | `null` | Exact manifest reference; required with an external `--model-path` |
| `--model-path` | path | `null` | External HBM path; omitted means the sample model path for the selected asset |
| `--test-clip` | path | `samples/vision/3dresnet/test_data/video0.npy` | Prepared `.npy` input |
| `--label-file` | path | `samples/vision/3dresnet/test_data/kinetics_classnames.json` | 400-entry source label mapping |
| `--top-k` | integer | `5` | Number of returned predictions, from 1 through 400 |
| `--priority` | integer | `0` | Runtime scheduling priority, 0 through 255 |
| `--bpu-cores` | one or more integers | `[0]` | Nonnegative BPU core indexes |
| `--list-models` | flag | `false` | Print the exact manifest asset without loading a model |
| `--dry-run` | flag | `false` | Print resolved paths and tensor contract without loading SDK |

An external path must be paired with `--asset-id s:3dresnet:s100/r3d_18.hbm`; the runtime never infers identity from a filename. `--dry-run` requires explicit `--target s100` on a host.

<a id="results"></a>
## Results

Successful CLI execution prints one JSON object to stdout:

```json
{
  "asset_id": "s:3dresnet:s100/r3d_18.hbm",
  "target": "s100",
  "clip": ".../test_data/video0.npy",
  "predictions": [
    {"class_id": 5, "score": "float32 probability", "label": "archery"}
  ]
}
```

`predictions` contains exactly `--top-k` entries sorted by source-compatible softmax probability. `class_id` is an integer in `[0,399]`; `score` is the float32 softmax value; `label` is the source JSON name with literal double quotes removed. No output file is written by the CLI.

<a id="integration-example"></a>
## Integration Example

The directory name `3dresnet` cannot appear in a `from ... import ...` statement. Run this example from the repository root and use `importlib.import_module` with the full package names; no runtime-directory `sys.path` injection is needed:

```python
import importlib
import sys
from pathlib import Path

import numpy as np

repo = Path.cwd()
binding_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
runner_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.model_runner")
task_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.classification")
labels_mod = importlib.import_module("samples.vision.3dresnet.runtime.python.labels")

selection = binding_mod.resolve_selection(
    "s100",
    asset_id="s:3dresnet:s100/r3d_18.hbm",
    model_path=repo / "samples/vision/3dresnet/model/s100/r3d_18.hbm",
)
runner = runner_mod.RuntimeModelRunner(selection)
binding = runner.load()
labels = labels_mod.load_labels(repo / "samples/vision/3dresnet/test_data/kinetics_classnames.json")
task = task_mod.VideoClassificationTask(runner, binding, top_k=5, labels=labels)
clip = np.load(repo / "samples/vision/3dresnet/test_data/video0.npy", allow_pickle=False)

prepared = task.pre_process(clip)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(clip)
assert np.array_equal(explicit_result.class_ids, composed_result.class_ids)
assert np.array_equal(explicit_result.scores, composed_result.scores)
assert explicit_result.labels == composed_result.labels
```

<a id="stage-io"></a>
## Four-stage API I/O

| Stage | Contract |
| --- | --- |
| `pre_process(clip)` | Input NumPy numeric clip of exact shape `(1,3,16,112,112)`; returns `PreparedInput.tensors` under the runtime-reported input name, contiguous float32 with the same values after casting, plus a per-call `VideoContext`. |
| `forward(tensors)` | Validates one actual input name, five-dimensional shape, F32 finite values, and one actual runtime output name; returns raw F32 scores without softmax or file I/O. |
| `post_process(outputs)` | Validates the actual output name, bound 400-score shape, F32 dtype, and finite values; delegates softmax/Top-K to `samples._shared.classification.topk_from_scores`. |
| `predict(clip)` | Runs the three stages in order and returns a `ClassificationResult`; it does not keep context in task state. |

The input clip is already RGB and normalized. The task performs no image or video decoding, resizing, or normalization.

<a id="troubleshooting"></a>
## Troubleshooting

- **Model path rejected:** provide the exact asset ID with an external path: `s:3dresnet:s100/r3d_18.hbm`.
- **Board identity rejected:** explicit `--target s100` selects the publication row but does not provide hardware evidence; execution still requires detected S100 identity.
- **Model missing:** run `bash samples/vision/3dresnet/model/download.sh s100` from the repository root.
- **Clip shape rejected:** use the prepared `.npy` clip with exact shape `(1,3,16,112,112)`; this runtime does not reshape or decode video.
- **Tensor name mismatch:** the binding reads the unique runtime names. It rejects missing, extra, or reordered tensors rather than inventing `input` or `output` names.
