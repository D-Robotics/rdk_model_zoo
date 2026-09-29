[English](README.md) | [简体中文](README_cn.md)

# LaneNet Python runtime

<a id="environment"></a>
## Environment

Real inference requires S100, its matching `hbm_runtime` Python runtime, NumPy and OpenCV. Host inspection (`--list-models`, `--dry-run`) does not import the board SDK or download a model. Host tests use explicitly injected SDK fixtures; they do not establish a supported board image or SDK version. The actual runtime version is recorded when available, otherwise `unknown`.

Prepare the HBM with [the explicit model downloader](../../model/README.md). The runtime does not install dependencies. Run all commands below from the repository root; the shell wrapper also changes to that root.

<a id="usage"></a>
## Usage

Inspect selection without execution:

```bash
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --dry-run
```

On an S100, run once with a new output directory:

```bash
bash samples/vision/lanenet/runtime/python/run.sh --target s100 --output outputs/lanenet_python
```

To use a prepared external model and image:

```bash
python3 -m samples.vision.lanenet.runtime.python.main --target s100 --asset-id s:lanenet:s100/lanenet256x512.hbm --model-path /data/models/lanenet256x512.hbm --test-img /data/road.jpg --output outputs/lanenet_external
```

The model path requires an explicit asset ID. This declares the intended contract; absent a publisher checksum it does not authenticate the file. Unsupported targets fail rather than selecting S100 silently.

<a id="parameters"></a>
## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--target` | `auto` | Resolves sole published S100 contract; execution separately checks physical identity |
| `--asset-id` | `null` | Inferred exact published identity; required for external model path |
| `--model-path` | `null` | Resolves sample `model/s100/lanenet256x512.hbm` |
| `--test-img` | `samples/vision/lanenet/test_data/lane.jpg` | OpenCV BGR input image |
| `--output` | `outputs/lanenet` | New result directory |
| `--instance-save-path` | `null` | Additional embedding display path; retained source option |
| `--binary-save-path` | `null` | Additional binary display path; retained source option |
| `--priority` | `0` | SDK priority, validated 0..255 |
| `--bpu-cores` | `[0]` | One or more nonnegative SDK core IDs; actual support depends on runtime |
| `--list-models` | `false` | List manifest records without execution |
| `--dry-run` | `false` | Print selected identity/path without execution |

The two inspection modes are mutually exclusive. Additional display files must be new and distinct and cannot overwrite canonical results. No implicit warmup or timing is performed. Source Python scheduling (priority 0, core 0) is retained; native scheduling uses the source UCP defaults instead.

<a id="results"></a>
## Results

| File | Meaning |
| --- | --- |
| `raw_outputs.npz` | All named outputs, exact dtype and shape; `report.json.raw_tensor_keys` maps SDK names to archive keys `output_N` |
| `embedding.npy` | Owned float32 `[3,256,512]` embedding, unchanged values |
| `binary.npy` | Owned uint8 `[256,512]` labels, restricted to 0/1 |
| `instance_pred.png` | Clipped/rounded embedding display; not clustered lane IDs |
| `binary_pred.png` | Labels rendered as 0/255 |
| `report.json` | Model/input digests, actual metadata, scheduling, processing boundaries and optional display paths |

Do not interpret NPZ keys as semantic output names: use the recorded map. Additional observed tensors remain raw even when the task does not consume them. The source prose's third output is unnamed and no third identity is invented here.

The display clips embeddings to [0,1], multiplies by 255, uses nearest ties-to-even rounding and preserves channel order. Source Python instead multiplied and cast to uint8, which truncated or wrapped out-of-range values. This intentional display change does not change `embedding.npy`. Binary labels outside 0/1 are rejected rather than displayed as plausible masks. Results stay at 256×512; no original-resolution interpolation or lane fitting occurs.

<a id="integration-example"></a>
## Application integration

Execute this from the repository root after preparing the default model on S100. It uses the same task and runner as the CLI, without rendering or writing result files:

```python
import cv2
from samples.vision.lanenet.runtime.python.model_binding import resolve_selection
from samples.vision.lanenet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.lanenet.runtime.python.lanenet import LaneNetTask

selection = resolve_selection("s100")
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = LaneNetTask(runner, binding)
image = cv2.imread("samples/vision/lanenet/test_data/lane.jpg", cv2.IMREAD_COLOR)
if image is None:
    raise ValueError("Cannot decode input image")
result = task.predict(image)
assert result.embedding.shape == (3, 256, 512)
assert result.binary.shape == (256, 512)
```

For raw output comparisons call `pre_process`, `forward`, then `post_process` separately and retain the `forward` mapping. Keep visualization and filesystem operations in the application. Do not share a mutable SDK runner across concurrent calls without an application-level synchronization policy.

<a id="stage-io"></a>
## Stage IO and binding

| Stage | Input | Output |
| --- | --- | --- |
| `pre_process` | Nonempty BGR uint8 HWC with 3 channels | Mapping from bound input name to contiguous float32 NCHW `[1,3,256,512]` |
| `forward` | Prepared mapping | All actual named raw output arrays, copied from runtime storage |
| `post_process` | Raw mapping matching metadata | `LaneResult`: CHW float embedding and HW uint8 binary labels |
| `predict` | BGR image | Composition of the three stages |

Preprocessing is source-compatible: BGR→RGB, INTER_AREA stretch to width 512/height 256, /255, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, CHW and batch. The same pure image function prepares calibration data. No letterbox, sigmoid, softmax, argmax or clustering is added.

Binding requires one model, one float32 `[1,3,256,512]` input, `instance_seg_logits` float32 `[1,3,256,512]`, and `binary_seg_pred` int64 `[1,1,256,512]` or `[1,256,512]`. Names, ranks, dtypes, finite values and all declared output shapes are checked. Auxiliary outputs may be float16/float32/int8/uint8/int16/int32/int64 with fixed positive shapes. Raw values are preserved; the task returns independent copies of its selected results. This Python auxiliary dtype set is broader than the current native set (no native float16), so validate actual metadata before cross-language comparison.

<a id="troubleshooting"></a>
## Troubleshooting and verification boundary

- Missing model: explicitly prepare it using the model instructions; `--dry-run` does not validate file contents.
- Identity rejection: verify the physical board. S100P/S600 names are not interchangeable with S100.
- Metadata mismatch: retain actual names/shapes/dtypes; do not bypass validation by renaming a tensor without establishing semantics.
- Invalid image or existing output path: choose a decodable image and a new output directory. A partial IO failure can leave an incomplete directory; inspect the error before reusing results.
- Unexpected colors: inspect raw embeddings separately; no instance clustering is implemented.

Host tests cover source preprocessing, binding, raw ownership, labels, displays and the actual CLI using a fake SDK. Board inference, real SDK compatibility, dataset accuracy and latency remain **not-run**. See [evaluation boundaries](../../evaluator/README.md) before reporting accuracy or equivalence.
