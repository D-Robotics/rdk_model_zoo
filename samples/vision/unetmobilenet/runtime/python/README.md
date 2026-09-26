English | [简体中文](README_cn.md)

# UNetMobileNet Python runtime

<a id="environment"></a>
## Environment

Python 3.10+ on S100/S600 with the matching board hbm_runtime. The source used NumPy 1.26.4/OpenCV 4.11.0.86; the canonical code additionally reads manifests with PyYAML and does not need source utility SciPy imports. Install general packages with `python3 -m pip install numpy opencv-python PyYAML`. No minimum S OS/SDK version was pinned in source; actual compatibility remains unverified here. Help/list/dry-run do not load SDK.

<a id="usage"></a>
## Usage

```bash
# cwd: repository root; run on the selected S100 board
bash samples/vision/unetmobilenet/model/download.sh --target s100
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100
# For S600, use --target s600 for BOTH preparation and inference.
# Host-only selection inspection, no SDK/model/download:
python3 samples/vision/unetmobilenet/runtime/python/main.py --dry-run --target s600
```

```bash
# cwd: repository root, recognized S board, model already prepared
python3 samples/vision/unetmobilenet/runtime/python/main.py
python3 samples/vision/unetmobilenet/runtime/python/main.py --target s100 --alpha-f 0.5 --img-save-path outputs/unetmobilenet/overlay.png --mask-save-path outputs/unetmobilenet/labels.npy --report-path outputs/unetmobilenet/report.json
```

run.sh sets cwd to the repository root and forwards arguments. It does not install/download. Success returns 0; failures return 2.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | exact board identity or explicit preparation target |
| `--asset-id` | str | `None` | exact target-scoped published identity |
| `--model-path` | Path | `None` | default model/<target>/ HBM; external copy requires asset-id |
| `--test-img` | Path | `samples/vision/unetmobilenet/test_data/segmentation.png` | sample-relative absolute default |
| `--img-save-path` | Path | `result.jpg` | original-resolution overlay |
| `--mask-save-path` | Path | `unetmobilenet_mask.npy` | original-resolution int32 labels; .npy required |
| `--report-path` | Path | `unetmobilenet_report.json` | JSON report |
| `--alpha-f` | float | `0.75` | ORIGINAL image weight in [0,1] |
| `--priority` | int | `0` | source priority, 0..255 |
| `--bpu-cores` | int list | `[0]` | source core list, nonnegative indexes |
| `--list-models` | flag | `false` | list manifest without SDK |
| `--dry-run` | flag | `false` | selection/config inspection; exclusive with list-models |

--help/-h prints help. Relative custom paths use cwd; existing outputs are replaced. Host dry-run needs an explicit target or exact asset-id. Auto on actual boards never defaults unknown/S100P identity to S100.

<a id="results"></a>
## Results

The NPY mask is original-resolution int32 IDs 0..18; result.jpg blends source colors at the same size. JSON/stdout records target, asset_id, model_path, input_path, publisher_sha256, runtime_version, metadata, mask_shape, class_ids, alpha_f, img_save_path and mask_save_path. No mIoU, confidence or latency is computed.

<a id="integration-example"></a>
## Integration example

```python
# cwd: repository root; on S100 after explicit model preparation
import cv2
from samples.vision.unetmobilenet.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.vision.unetmobilenet.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.unetmobilenet.runtime.python.unetmobilenet import UnetMobileNetTask
from samples.vision.unetmobilenet.runtime.python.visualization import render_overlay

image = cv2.imread(str(SAMPLE_DIR / "test_data/segmentation.png"))
runner = RuntimeModelRunner(resolve_selection("s100"))
binding = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
task = UnetMobileNetTask(runner, binding)
prepared = task.pre_process(image)
raw = task.forward(prepared.tensors)
mask = task.post_process(raw, prepared.context)
mask_again = task.predict(image)
overlay = render_overlay(image, mask, alpha_f=0.75)
print(mask.shape, mask.dtype)  # original image height/width, int32
```
Unlike the archived UnetMobileNet.predict API, task.predict returns class IDs; call render_overlay explicitly. Runner owns SDK lifetime and scheduling, binding owns exact artifact/metadata contracts. No shared mutable “last image size” is stored on the task; retain each PreparedInput context. SDK concurrency is not guaranteed.

<a id="stage-io"></a>
## Stage IO

pre_process accepts nonempty BGR uint8 HWC, stretches with INTER_AREA to 2048×1024, creates Y uint8 [1,1024,2048,1] and UV uint8 [1,512,1024,2], and freezes original geometry per call. No CPU normalization or letterbox. forward returns raw [1,H,W,19] int32/F32 unchanged. post_process accepts bound geometry/dtype and finite values. Explicit NONE int32 scores compare without float rounding; SCALE uses validated positive scales/offsets and float64 affine decoding, fixing the source assumption that raw integer argmax always preserves order. F32 is not dequantized again. Lowest class ID wins exact ties. IDs resize directly to original dimensions with INTER_NEAREST; no coloring or file IO occurs. Missing integer quantization metadata fails explicitly.

<a id="troubleshooting"></a>
## Troubleshooting

Missing model: use explicit download for the same target. Unknown board: use explicit target for host inspection only; actual execution still checks identity. Wrong output channels/dtype/quantization: inspect real metadata rather than renaming/reinterpreting the model. Colors differ from Cityscapes standard palette by design: this preserves source rdk_colors. alpha_f weights original pixels, not mask. Original-size mask is not the fixed 512×512 output of the separate X5 UNet sample.
