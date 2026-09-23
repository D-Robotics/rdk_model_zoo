# Python Runtime — FCOS

<a id="environment"></a>
## Environment

- Board: RDK X5 board image with `hbm_runtime`; this board runtime is not imported by host help/list/dry-run.
- Host: Python 3.10+ with `numpy`, `opencv-python`, and `PyYAML` from `requirements-host.txt`.
- The runner loads the SDK only after target identity, exact asset identity, and model-file checks pass.

<a id="usage"></a>
## Usage

```bash
# cwd: repository root; artifact must already be prepared by model/download.sh
python3 samples/vision/fcos/runtime/python/main.py --target x5 --asset-id x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
# success: exit 0, JSON result and test_data/result.jpg are produced on the board

# cwd: repository root; explicit B3 selection
python3 samples/vision/fcos/runtime/python/main.py \
  --target x5 --variant efficientnetb3 \
  --asset-id x5:fcos:fcos_efficientnetb3_detect_896x896_bayese_nv12.bin \
  --test-img samples/vision/fcos/test_data/bus.jpg --img-save-path /tmp/fcos-b3.jpg
# success: exit 0 and /tmp/fcos-b3.jpg exists
```

`run.sh` only delegates to `main.py`; it never downloads or installs anything.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | `x5` for execution; `auto` is listing-only |
| `--asset-id` | str | `None` | Exact `x5:fcos:<filename>` identity |
| `--variant` | str | `None` | Omitted selects B0; with an asset ID it may select B0/B2/B3 |
| `--model-path` | str | `None` | External file; requires exact asset ID |
| `--test-img` | str | `samples/vision/fcos/test_data/bus.jpg` | BGR input image |
| `--label-file` | str | `datasets/coco/coco_classes.names` | Optional COCO names |
| `--img-save-path` | str | `samples/vision/fcos/test_data/result.jpg` | Annotated JPEG |
| `--resize-type` | int | `None` | Source default 0 direct resize; 1 letterbox with inverse padding/scale restoration |
| `--classes-num` | int | `80` | Source FCOS class count; binding remains 80 |
| `--conf-thres` | float | `0.5` | FCOS confidence threshold |
| `--iou-thres` | float | `0.6` | OpenCV NMS IoU threshold |
| `--priority` | int | `0` | Runtime scheduling priority |
| `--bpu-cores` | int list | `[0]` | Runtime BPU cores |
| `--list-models` | flag | `false` | List three assets without SDK |
| `--dry-run` | flag | `false` | Resolve selection and protocol without SDK |

<a id="results"></a>
## Results

Stdout is JSON with `asset_id`, `boxes`, `scores`, `class_ids`, and `result_path`. `boxes` are owned float32 `[x1,y1,x2,y2]` pixel coordinates clipped to the original BGR image, `scores` are float32 source FCOS confidence, and `class_ids` are owned int32 zero-based class IDs. The image path receives rectangles and optional labels.

<a id="integration-example"></a>
## Integration Example

Prerequisite: prepare the exact B0 artifact with `model/download.sh`; `bus.jpg` is bundled.

```python
import cv2
import numpy as np
from samples.vision.fcos.runtime.python.fcos import FCOSTask
from samples.vision.fcos.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.fcos.runtime.python.model_runner import RuntimeModelRunner
from samples._shared.runtime_meta import RuntimeMetadata

selection = resolve_selection("x5", asset_id="x5:fcos:fcos_efficientnetb0_detect_512x512_bayese_nv12.bin")
runner = RuntimeModelRunner(selection)
binding = runner.load()  # board image only; host tests inject a runtime
task = FCOSTask(runner, binding)
image = cv2.imread("samples/vision/fcos/test_data/bus.jpg", cv2.IMREAD_COLOR)
result = task.predict(np.asarray(image))
print(result.boxes, result.scores, result.class_ids)
```

<a id="stage-io"></a>
## Three-Stage I/O

- `pre_process`: BGR `uint8 (H,W,3)` → packed contiguous NV12 `uint8 (1.5*input_h*input_w,)` plus frozen `ImageContext`.
- `forward`: validated packed tensor → raw mapping of 15 arrays; output shapes are `(1,input_h/stride,input_w/stride,{80,4,1})`; no activation, dequantization, NMS, or file I/O.
- `post_process`: raw mapping + context → dequantized FCOS confidence `sqrt(sigmoid(cls_max)*sigmoid(center))`, stride-scaled `xyxy`, source OpenCV NMS, and original-image result. Direct resize uses independent width/height ratios; letterbox subtracts the frozen top/left padding and divides by the effective integer resized width/height before clipping.
- Quantization: FCOS follows the fixed source `dequantize_outputs` path. A SCALE descriptor is applied even when the observed runtime array is F32; every output must carry an inspectable descriptor. Missing or unknown descriptors are rejected, and dtype alone never selects a raw-F32 path.
- `predict` calls the three stages exactly once in that order for each image.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `--model-path requires the exact --asset-id reference` | External identity is incomplete | Pass the full manifest reference from `--list-models` |
| `Local execution requires recognized board identity` | Host or wrong board reached the real runner | Use `--help`, `--list-models`, or `--dry-run` on host; run inference on matching X5 |
| `Expected exactly one FCOS output with shape ...` | Artifact metadata does not match the selected resolution | Use the matching variant/asset and do not infer protocol from a filename |
