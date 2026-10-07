# YOLOE Python runtime

<a id="environment"></a>
## Environment

Python 3.10+, NumPy/OpenCV/SciPy/PyYAML. Only actual model execution imports board `hbm_runtime`. See [root prerequisites](../../README.md#prerequisites) for recorded dependency versions and boundaries. Published S quantized-output models require a local float conversion to run through this entry.

<a id="usage"></a>
## Usage

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11s
```

Zero-argument execution detects the local board and derives the default variant. X5 requires prior model preparation; S reports the float-artifact gap. Customized X5 example:

```bash
# cwd: repository root; first prepare 11m using model/download.sh --target x5 --variant 11m
python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11m --score-thres 0.35 --resize-type 0
```

Success is rc=0 with JSON detection count/classes/scores/result path; errors return rc=2. Dry-run validates selection and the tensor contract; SDK compatibility is exercised by real inference.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | Execution target; auto reads local identity |
| `--variant` | str | `null` | 11s/m/l or 26n/s/m/l/x; target-specific default |
| `--asset-id` | str | `null` | Exact source publication ID |
| `--model-path` | str | `null` | Original publication or separately converted float file |
| `--local-float-sha256` | str | `null` | Local float conversion digest; requires model path |
| `--test-img` | str | `samples/vision/yoloe/test_data/office_desk.jpg` | Input image decoded as BGR |
| `--label-file` | str | `samples/vision/yoloe/test_data/classes.names` | Hash-pinned ordered 4585-class vocabulary |
| `--img-save-path` | str | `samples/vision/yoloe/test_data/result.jpg` | Output image with boxes and masks |
| `--score-thres` | float | `0.25` | Strict sigmoid confidence threshold |
| `--nms-thres` | float | `null` | 11 default 0.7; forbidden for 26 |
| `--resize-type` | int | `1` | 11: 0 stretch/1 letterbox; 26 requires 1 |
| `--no-morph` | flag | `false` | Disable S11 CLI default mask opening |
| `--no-contour` | flag | `false` | Disable mask contour outlines |
| `--max-det` | int | `300` | 26 Top-K count 1..8400; 11 is not truncated |
| `--multi-label` | flag | `false` | 26 only: multiple classes per anchor |
| `--priority` | int | `0` | Scheduling priority 0..255 |
| `--bpu-cores` | int | `[0]` | Nonnegative core IDs; board-specific validity is checked by SDK |
| `--list-models` | flag | `false` | List publication matrix without SDK |
| `--dry-run` | flag | `false` | Resolve selection without loading; explicit target required |

X5 11 clamps confidence to `[1e-6,1-1e-6]` before logit conversion; S11/26 use the supplied threshold directly.

<a id="results"></a>
## Results

`Result.boxes` is float32 `[N,4]` continuous original-image xyxy pixels, clipped to `[0,W]/[0,H]`; `scores` is float32 `[N]` sigmoid probability and `class_ids` is int64 `[N]` fixed-vocabulary ID, not COCO category ID. X5 `masks` is bool `[N,H,W]` (`mask_layout="full"`); S returns N uint8 0/1 ROI arrays (`mask_layout="roi"`), sliced using integer-truncated box bounds with empty ROI alignment retained. Returned results own their memory. S11 retains exact zero-axis ROI shapes and normalizes Lanczos overshoot to binary 0/1 without changing foreground support.

The CLI saves a colored overlay, default `test_data/result.jpg`. It does not save raw tensors or present inference as an accuracy report. The CLI entry is organized as: `main.py` resolves the selection, builds the `Config`, constructs `YOLOE` with its runner, calls `predict` once and renders the result; option declarations, the `--list-models`/`--dry-run` modes and the JSON result report live in `cli.py`. The segmentation implementation lives in `yoloe.py` with its decode/IO modules.

<a id="integration-example"></a>
## Integration Example

Prerequisites: a Python 3.10+ environment and the 11s model explicitly prepared on X5 with `model/download.sh`; on-board runs load the board `hbm_runtime` SDK.

```python
# cwd: repository root; on X5 after the explicit model/download.sh step
from samples.vision.yoloe.runtime.python.model_binding import resolve_selection, SAMPLE_DIR
from samples.vision.yoloe.runtime.python.model_runner import build_runner
from samples.vision.yoloe.runtime.python.visualization import load_inputs
from samples.vision.yoloe.runtime.python.yoloe import YOLOE, Config
selection = resolve_selection("x5", variant="11s")
runner = build_runner(selection)
image, labels = load_inputs(SAMPLE_DIR / "test_data/office_desk.jpg", SAMPLE_DIR / "test_data/classes.names")
task = YOLOE(selection, Config(), runner=runner)
result = task.predict(image)
print(result.boxes.shape, result.mask_layout)
```

Configuration and validation live in `config.py`, shared with the native launcher without importing image or SDK modules. `from...yoloe import Config` is also supported.

<a id="stage-io"></a>
## Three-Stage I/O

`YOLOE(selection, Config, runner=...)` constructs the task; Config is frozen. The stages are spelled `preprocess` / `infer` / `postprocess`, with the established `pre_process` / `forward` / `post_process` names as thin aliases (one implementation per stage). preprocess accepts nonempty uint8 BGR HWC and returns `Prepared.tensors` plus this image’s `context`. X5 sends 614400 packed NV12 bytes as a 1D array. S sends Y `[1,640,640,1]` and UV `[1,320,320,2]`. Version 11 truncates resize dimensions and pads 127 (stretch uses nearest); 26 rounds dimensions and pads 114.

infer calls the runner once, preserving raw float32 without activation/dequantization. `RawOutputs` borrows SDK arrays: consume before the next call, or copy explicitly. Each stride 8/16/32 provides cls 4585, box 64 (11) or 4 (26), and mces 32 channels, plus NHWC `[1,160,160,32]` proto. Complete shapes uniquely bind actual outputs; names/enumeration order are not assumptions.

postprocess requires matching context. Version 11 uses DFL and NMS: X5 crops low-resolution mask probabilities before two linear resizes; S uses binary ROI masks. Version 26 interpolates logits at 640 before binarization, unpadding and nearest restoration. Boxes use actual per-axis integer resize scales. predict only composes stages, with no last-image state or SDK thread-safety promise.

When intermediate tensors are needed, call the three stages explicitly; this is equivalent to `predict` and still runs inference exactly once:

```python
prepared = task.preprocess(image)
raw = task.infer(prepared)
result = task.postprocess(raw, prepared.context)
```

Library Config defaults do_morph=False, preserving S11 library behavior; the CLI defaults to True for S11, preserving the source CLI. Scheduling is exposed through `runner.set_scheduling_params`.

X5 accepts RGB-shaped metadata layouts `[1,3,640,640]` and `[1,640,640,3]`; either still transports 614400 packed NV12 bytes. The shared binder enables NHWC RGB metadata only for the explicit YOLOE-11 contract. It does not relax the other samples’ input contracts.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Cause and action |
| --- | --- |
| `Published S YOLOE outputs are quantized` | Original S HBM cannot enter the float route; retain Dequantize output nodes in a separate conversion, record its new digest and validate SDK descriptors. |
| `Local float SHA-256 mismatch` | Local bytes differ from the supplied digest; verify the conversion output, never reuse the original publication hash. |
| `No unique YOLOE asset` | Conflicting target/variant/asset-id or absent asset; inspect --list-models, never rename a file to impersonate a target. |
| `Vocabulary checksum mismatch` | Restore matching ordered labels; relabeling does not change learned classes. |
| `YOLOE requires the ten declared NHWC float32 outputs` | Wrong export, layout or precision; inspect conversion rather than casting integers. |

For a separate float conversion, start with the [conversion preparation guide](../../conversion/README.md); calibration, compile success and verified output precision are reported separately.
