English | [简体中文](README_cn.md)

# UNet Python runtime

<a id="overview"></a>
## Python inference

Generate a 21-class VOC segmentation mask with the selected U-Net backbone.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── cli.py  # Arguments, model selection and result output
├── main.py  # CLI entry: construct model and call predict
├── run.sh  # Run the sample
└── unet.py  # Model initialization and inference stages
```

`main.py` constructs the model and calls `predict`; `cli.py` handles options, model selection, and image/report output. The task file owns model stages and image loading uses `utils.py_utils.image.read_bgr_image`.

Start with [main.py](main.py): it constructs `UNetSegmenter` and calls `predict`. [unet.py](unet.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

RDK X5, OS 3.5.0+, Python 3.10+, and board-provided hbm_runtime; do not substitute an arbitrary same-named PyPI package. Host help/list/dry-run need no SDK or model.

```bash
# cwd: repository root; general Python dependencies only
python3 -m pip install numpy opencv-python PyYAML
python3 samples/vision/unet/runtime/python/main.py --help
```

<a id="usage"></a>
## Usage

Run below from the repository root. run.sh only forwards arguments and changes cwd to the root; it never downloads. Successful inference returns 0, errors return 2.
```bash
# cwd: repository root; explicitly prepare the selected artifact first
bash samples/vision/unet/model/download.sh --target x5 --variant resnet18
python3 samples/vision/unet/runtime/python/main.py
python3 samples/vision/unet/runtime/python/main.py --target x5 --variant resnet18 --test-img samples/vision/unet/test_data/2007_000033.jpg --alpha 0.4 --mask-save-path outputs/unet/mask.png --img-save-path outputs/unet/overlay.png --report-path outputs/unet/report.json
python3 samples/vision/unet/runtime/python/main.py --dry-run --target x5 --variant resnet34
```

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | choice | `auto` | auto resolves x5; real execution verifies local identity |
| `--variant` | choice | `None` | resnet18 by default, or inferred from exact asset-id |
| `--asset-id` | str | `None` | x5:unet:<filename>; must agree with variant |
| `--model-path` | Path | `None` | default selected file under model/; external copy requires asset-id |
| `--test-img` | Path | `samples/vision/unet/test_data/2007_000033.jpg` | absolute sample-relative default; BGR image |
| `--mask-save-path` | Path | `unet_mask.png` | class-ID PNG |
| `--img-save-path` | Path | `unet_result.png` | color overlay |
| `--report-path` | Path | `unet_runtime_report.json` | JSON report |
| `--priority` | int | `None` | 0..255; omit to keep SDK default |
| `--bpu-core` | int | `None` | nonnegative SDK core index; omit to keep default |
| `--alpha` | float | `0.55` | overlay weight in [0,1] |
| `--dry-run` | flag | `false` | resolve only, no model load/download |
| `--list-models` | flag | `false` | manifest rows only; exclusive with dry-run |

`--help` / `-h` prints help. Relative output paths use cwd; existing results are replaced. `--dry-run` prints the selected model paths and CLI configuration.

<a id="results"></a>
## Results

The mask is fixed 512×512 uint8, VOC IDs 0..20. Overlay uses the same size and alpha-blends resized input with the VOC palette; original-size restoration is not automatic. JSON includes target, variant, asset_id, runtime_version, model_path, image_path, metadata, mask_shape, classes_present, elapsed_ms and output paths. Timing covers all three stages, including preprocessing and postprocessing.

<a id="integration-example"></a>
## Integration

```python
# cwd: repository root; on X5 after explicit model preparation
import cv2
from samples.vision.unet.runtime.python.cli import resolve_selection, SAMPLE_DIR
from samples.vision.unet.runtime.python.unet import UNetSegmenter

image = cv2.imread(str(SAMPLE_DIR / "test_data/2007_000033.jpg"))
selection = resolve_selection("x5", variant="resnet18")
task = UNetSegmenter(selection)
prepared = task.preprocess(image)
raw = task.infer(prepared.tensors)
mask = task.postprocess(raw)
mask_again = task.predict(image)
print(mask.shape, mask.dtype)  # (512, 512), uint8
```
The model initializes the runtime and implements preprocessing, inference, postprocessing and `predict`. Use a separate model instance for each concurrent worker.

<a id="stage-io"></a>
## 阶段 I/O / Stage IO

Preprocess accepts nonempty BGR uint8 HWC, stretches with INTER_LINEAR to 512×512, then creates contiguous packed NV12 uint8 `(1,768,512,1)`. Metadata may describe logical NCHW `(1,3,512,512)`, NHWC `(1,512,512,3)` or physical packed shape, with NV12 dtype. Each call returns independent frozen original-size context.

infer only returns runner-validated raw logits, with no dequantization or argmax. postprocess accepts `(1,21,512,512)` or `(1,512,512,21)`. Integer outputs require valid SCALE parameters; float32 is not dequantized again even with a vestigial descriptor. Class argmax breaks ties by lowest ID. Model-resolution output needs no context restoration; no softmax, automatic resize-back or file IO occurs.

<a id="troubleshooting"></a>
## Troubleshooting

Missing model: prepare it explicitly. Hash mismatch: verify the published file instead of bypassing checks. Locally compiled files use evaluator --model, with --backbone for custom filenames. S100/S100P/S600 have no assets here and cannot fall back to X5. Images must be nonempty uint8 with three channels; investigate incompatible exports/artifacts rather than changing class count or size to force execution.
