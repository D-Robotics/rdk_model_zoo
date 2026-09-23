# YOLOv5 Python runtime

<a id="environment"></a>
## Environment

Use Python 3 with NumPy and OpenCV. Board inference additionally needs the matching `hbm_runtime`; S post-processing uses the declared quantization descriptors and may need the host CPU packages listed by the sample. `--help`, `--list-models`, and explicit-target `--dry-run` do not load the SDK, download models, or read board identity. X5 binds one packed uint8 NV12 tensor `(640*640*3/2,)`; S binds Y `(1,672,672,1)` and UV `(1,336,336,2)`.

<a id="usage"></a>
## Usage

From the repository root, after the explicit model preparation in `model/README.md`:

```bash
python3 -m samples.vision.yolov5.runtime.python.main --target x5 --variant n-v7.0
```

This writes `samples/vision/yolov5/test_data/result_unified.jpg` and prints JSON detection arrays; exit code `0` is success. A custom S invocation is:

```bash
python3 -m samples.vision.yolov5.runtime.python.main \
  --target s100 --variant x-672 \
  --asset-id s:yolov5:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/yolov5/model/s100/yolov5x_672x672_nv12.hbm \
  --test-img samples/vision/yolov5/test_data/kite.jpg
```

`bash samples/vision/yolov5/runtime/python/run.sh ...` is the same module entry. The S command is conditional on a prepared model and recognized S100 board; neither was used here.

<a id="parameters"></a>
## Parameters

| option | type | default | meaning |
|---|---|---|---|
| `--target` | choice | `auto` | `auto`, `x5`, `s100`, `s100p`, or `s600`; execution still requires a supported asset |
| `--variant` | string | `null` | effective `n-v7.0` for X5 and `x-672` for S |
| `--asset-id` | string | `null` | exact manifest reference for a custom path |
| `--model-path` | path | `null` | existing model; requires matching asset ID |
| `--test-img` | path | `null` | effective X5 `test_data/bus.jpg`, S `test_data/kite.jpg` |
| `--label-file` | path | `samples/vision/yolov5/test_data/coco_classes.names` | visualization labels |
| `--img-save-path` | path | `samples/vision/yolov5/test_data/result_unified.jpg` | annotated output image |
| `--score-thres` | float | `0.25` | confidence threshold in [0,1] |
| `--nms-thres` | float | `0.45` | IoU threshold in [0,1] |
| `--resize-type` | int choice | `null` | effective X5 `0` stretch or S `1` letterbox |
| `--classes-num` | int choice | `80` | published contract is exactly 80 |
| `--anchors` | comma float list | `[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]` | 9 anchor pairs |
| `--strides` | comma int list | `[8, 16, 32]` | three output strides |
| `--priority` | int | `0` | scheduler priority 0..255 |
| `--bpu-cores` | int list | `[0]` | nonnegative BPU indexes |
| `--list-models` | flag | `false` | print manifest references without SDK |
| `--dry-run` | flag | `false` | explicit target only; print selection/protocol without SDK |

`--list-models` and `--dry-run` are mutually exclusive. Invalid user/runtime conditions return `2`.

<a id="results"></a>
## Results

The CLI prints `boxes`, `scores`, and `class_ids`, then saves the image path given by `--img-save-path`. `YOLOv5Task.forward` returns native output arrays without dequantization or reshaping; `post_process` applies S metadata dequantization when required, sigmoid/anchor decoding, thresholding, NMS, and inverse geometry. X5 intentionally preserves the source quirk of passing XYXY boxes to OpenCV `NMSBoxes` (which treats the list as XYWH); S uses class-wise XYXY NMS. This visible difference is part of the target protocol.

<a id="integration-example"></a>
## Integration example

After a matching model is prepared and the board target is recognized, this complete example defines all variables and compares explicit stages with `predict`:

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.yolov5.runtime.python.model_binding import resolve_selection
from samples.vision.yolov5.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.yolov5.runtime.python.detection import YOLOv5Task

target = "x5"
variant = "n-v7.0"
asset_id = "x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin"
model_path = Path("samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin")
image_path = Path("samples/vision/yolov5/test_data/bus.jpg")
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError(image_path)
selection = resolve_selection(target, variant=variant, asset_id=asset_id, model_path=model_path)
runner = RuntimeModelRunner(selection)
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = YOLOv5Task(runner, binding, score_thres=0.25, nms_thres=0.45)
prepared = task.pre_process(image)
native_outputs = task.forward(prepared.tensors)
explicit_result = task.post_process(native_outputs, prepared.context)
composed_result = task.predict(image)
for key in ("boxes", "scores", "class_ids"):
    assert np.array_equal(getattr(explicit_result, key), getattr(composed_result, key))
print(explicit_result.boxes, explicit_result.scores, explicit_result.class_ids)
```

<a id="stage-io"></a>
## Stage I/O

- `pre_process(image, resize_type=None)` accepts HWC uint8 BGR and returns `PreparedInput(tensors, context)`. It converts to target NV12 and stores original H/W and resize mode in immutable context.
- `forward(tensors)` validates names, shape, dtype, and finiteness, then returns native output arrays. It does not decode, dequantize, reshape, or mutate them.
- `post_process(outputs, context, score_thres=None, nms_thres=None)` consumes the matching context and returns owned `DetectionResult(boxes, scores, class_ids)`.
- `predict` composes the same stages. Context is per call; zero threshold overrides are preserved.

<a id="troubleshooting"></a>
## Troubleshooting

- A custom path without its exact manifest `--asset-id` is rejected.
- X5 `--variant` must be one of the nine published tags; S accepts only `x-672` on S100/S600.
- Wrong native output shape/dtype/quantization metadata is rejected before inference; no name-based layout guessing is used.
- X5 and S use different input containers and resize defaults. Do not feed packed X5 tensors to S or split S tensors to X5.
