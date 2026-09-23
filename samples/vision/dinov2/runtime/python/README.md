English | [简体中文](./README_cn.md)

# Python Runtime — DINOv2 vision features

<a id="environment"></a>
## Environment

- Board targets: RDK S100 (`nash-e`), S100P (`nash-m`), and S600 (`nash-p`), with a board image providing `hbm_runtime`. Board image and firmware versions were not verified in this migration.
- Host contract checks: Python 3.14.7 with `numpy`, `opencv-python`, and `PyYAML` from `../../requirements-host.txt`. Host `--help`, `--list-models`, and explicit-target `--dry-run` do not import or load the SDK.
- The runtime uses one target-specific HBM with both `cls_feat` and `patch_feat` outputs. Runtime instances and the installed SDK are not declared thread-safe.

<a id="usage"></a>
## Usage

Prepare an artifact with [`../../model/README.md`](../../model/README.md), then run on the target board from the repository root:

```bash
# cwd: repository root; model: target-specific HBM already prepared under samples/vision/dinov2/model/
python3 samples/vision/dinov2/runtime/python/main.py
# success: exit code 0, JSON summary for cls_feat and cosine similarity for bus.jpg
```

Select patch features and save the complete returned tensor at an exact path:

```bash
# cwd: repository root; model: prepared S100P artifact; inputs: bundled dog.jpg and bus.jpg
python3 samples/vision/dinov2/runtime/python/main.py \
  --target s100p --output patch_feat \
  --output-file /tmp/dinov2-patch.npy
# success: exit code 0, JSON summary with shape [1,256,384], and /tmp/dinov2-patch.npy; no extension is appended
```

`run.sh` accepts the historical positional output (`cls_feat` or `patch_feat`) followed by named options and does not download. `--list-models` and explicit-target `--dry-run` work without SDK loading; host dry-run with `--target auto` exits 2.

<a id="parameters"></a>
## Parameters

`--list-models` and `--dry-run` are mutually exclusive. Invalid selection, target, input, or artifact errors exit 2.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | `auto`, `s100`, `s100p`, or `s600`; `auto` detects the concrete board only for execution. |
| `--asset-id` | str | `None` | Exact qualified manifest reference, for example `s:dinov2:nash-e/dinov2_vits14_224_int16_nashe.hbm`. |
| `--model-path` | str | `None` | Explicit local HBM path; must be paired with the exact `--asset-id`. |
| `--test-img` | str | `samples/vision/dinov2/test_data/dog.jpg` | First BGR image path; parser resolves the bundled default from the sample file. |
| `--second-img` | str | `samples/vision/dinov2/test_data/bus.jpg` | Optional second BGR image used for cosine similarity; a missing file is reported and does not fail the run. |
| `--output` | str | `cls_feat` | Feature returned and summarized: `cls_feat` or `patch_feat`. |
| `--priority` | int | `0` | Runtime priority, constrained to 0–255. |
| `--bpu-cores` | int list | `[0]` | One or more nonnegative BPU core indexes. |
| `--output-file` | str | `None` | Optional exact path for the returned float32 NumPy array; parent directories are created and no extension is added. |
| `--list-models` | flag | `false` | List exact manifest assets without model loading. |
| `--dry-run` | flag | `false` | Resolve one explicit target and print the input/output source contract without SDK loading. |

<a id="results"></a>
## Results

The CLI prints JSON fields `output`, `shape`, `dtype`, `mean`, `std`, `min`, `max`, and `l2_norm`. If the second image is used, it also prints `second_image` and `cosine_similarity`; a missing file has status `skipped_missing`. `cls_feat` is `(1,384)`, `patch_feat` is `(1,256,384)`, and both results are owned float32 arrays. Integer HBM outputs are dequantized by the output quantization metadata. No softmax, L2 normalization, patch pooling, or other feature transformation is applied.

<a id="integration-example"></a>
## Integration Example

Prerequisite: prepare the S100 artifact using [`../../model/README.md`](../../model/README.md), and run this snippet on an S100 board. Change `target = "s100"` to `"s100p"` or `"s600"` when using those boards; selection then resolves that target's independent manifest HBM. The example defines all paths, input, target, asset identity, scheduling values, runner, binding, task, `explicit_result`, and `composed_result`. `DINOv2Task.post_process` returns the selected ndarray; repeat the same block with `output="patch_feat"` to compare the other output key.

```python
from pathlib import Path
import cv2
import numpy as np

from samples.vision.dinov2.runtime.python.embedding import DINOv2Task
from samples.vision.dinov2.runtime.python.model_binding import resolve_selection
from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner

repo = Path.cwd()
image_path = repo / "samples/vision/dinov2/test_data/dog.jpg"
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f"cannot read {image_path}")

target = "s100"
asset_id = None
model_path = None
priority = 0
bpu_cores = [0]
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)
output = "cls_feat"
task = DINOv2Task(runner, binding, output)
prepared = task.pre_process(image)
raw_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(raw_outputs)
composed_result = task.predict(image)
np.testing.assert_array_equal(explicit_result, composed_result)
print({"output": output, "shape": composed_result.shape,
       "dtype": str(composed_result.dtype)})
```

<a id="stage-io"></a>
## Three-Stage I/O

- `pre_process`: BGR `uint8` `H×W×3` → `PreparedInput`; OpenCV converts RGB, bicubic-resizes the short side to 256, center-crops 224, scales by `/255`, applies ImageNet mean/std, and emits owned contiguous float32 `{"input": (1,3,224,224)}`. `context` records original/resized shape and crop origin.
- `forward`: input mapping → raw mapping containing exactly `cls_feat` `(1,384)` and `patch_feat` `(1,256,384)`. The runner validates names, shape, and native metadata dtype and returns raw values unchanged.
- `post_process`: raw dual-output mapping → the selected output as an owned float32 ndarray. Float32 output stays raw; integer output uses only its bound quantization metadata for dequantization. No softmax or L2 normalization is applied.
- `predict(image)` composes the three stages for the task's selected output. It does not download, write files, or evaluate.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Model not found: ...; prepare it explicitly with model/download.sh.` | Target-specific HBM is absent. | Run `model/download.py --target ...` or pass an exact asset ID and model path. |
| `No published DINOv2 support for x5` | Target is outside the three published S targets. | Use `s100`, `s100p`, or `s600`. |
| `Host dry-run requires --target s100, s100p, or s600` | Host cannot infer a board for dry-run. | Supply an explicit target. |
| `An external model-path requires the exact manifest asset-id.` | A custom path was supplied without publication identity. | Add the matching `--asset-id`. |
| `DINOv2 ... differs from bound metadata` | HBM input/output metadata does not match the fixed contract. | Use the exact target artifact; do not reshape or cast tensors. |

## License

Runtime code follows the repository [LICENSE](../../../../../LICENSE), Apache-2.0.
