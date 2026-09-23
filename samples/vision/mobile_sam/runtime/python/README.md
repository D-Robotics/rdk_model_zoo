English | [简体中文](README_cn.md)

# MobileSAM Python Runtime

<a id="environment"></a>
## Environment

Host checks use the repository `.venv` with Python, NumPy, OpenCV and PyYAML for manifest reads; the recorded host fixture is Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0 and PyYAML 6.0.3. This sample does not pin NumPy/OpenCV versions. Runtime syntax requires Python 3.10 or newer; board SDK/system versions are unknown and not-run. `--list-models` and explicit-target `--dry-run` resolve the manifest and tensor contract without constructing the SDK.

The board's SDK must already be installed by its matching system image; do not install `hbm_runtime` from an unrelated host environment. Check the required Python imports from the repository root:

```bash
# cwd: repository root on the selected board
python3 -c "import numpy, cv2, yaml, hbm_runtime; print('runtime dependencies available')"
```

If only the ordinary Python dependencies are missing, install them in the Python environment used by that board's SDK (`python3 -m pip install numpy opencv-python PyYAML`). The source does not pin their board versions; preserve the image's SDK compatibility constraints. The command above checks import availability only. Disk/RAM requirements were not measured; both encoder and decoder must fit in the target runtime.

<a id="usage"></a>
## Usage

From the repository root, prepare the default pair for the detected board and run:

```bash
# cwd: repository root; prerequisite: default pair and matching board runtime
python3 samples/vision/mobile_sam/runtime/python/main.py
# expect: exit code 0, JSON on stdout, overlay and binary mask at the requested paths
```

For a fully explicit pair, box and output locations, use the following S100 command:

```bash
python3 samples/vision/mobile_sam/runtime/python/main.py \
  --target s100 \
  --encoder-asset-id s:mobile_sam:nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm \
  --decoder-asset-id s:mobile_sam:nash-e/mobile_sam_decoder_512_nashe.hbm \
  --encoder-model-path samples/vision/mobile_sam/model/nash-e/mobile_sam_image_encoder_norm_512x512_nashe.hbm \
  --decoder-model-path samples/vision/mobile_sam/model/nash-e/mobile_sam_decoder_512_nashe.hbm \
  --test-img samples/vision/mobile_sam/test_data/dogs.jpg \
  --img-save-path /absolute/mobile-overlay.jpg \
  --mask-save-path /absolute/mobile-mask.png \
  --box 185,120,380,445 --priority 0 --bpu-cores 0
```

`run.sh` delegates to this CLI and never downloads. The box is expressed in the resized `512x512` image coordinates. A custom model path requires the exact corresponding manifest asset ID.

<a id="parameters"></a>
## Parameters

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `--target` | choice | `auto` | `x5`, `s100`, `s100p`, or `s600` target |
| `--encoder-asset-id` | string | `null` | Exact encoder manifest reference |
| `--decoder-asset-id` | string | `null` | Exact decoder manifest reference |
| `--encoder-model-path` | path | `null` | External encoder path; requires encoder ID |
| `--decoder-model-path` | path | `null` | External decoder path; requires decoder ID |
| `--test-img` | path | samples/vision/mobile_sam/test_data/dogs.jpg | BGR input image |
| `--img-save-path` | path | samples/vision/mobile_sam/test_data/mobile_sam_full_mask_result.jpg | Overlay output |
| `--mask-save-path` | path | samples/vision/mobile_sam/test_data/mobile_sam_binary_mask_result.png | Binary mask output |
| `--box` | `x1,y1,x2,y2` | `[185.0, 120.0, 380.0, 445.0]` | Ordered box in 512-image coordinates |
| `--priority` | integer | `0` | Runtime scheduling priority |
| `--bpu-cores` | integers | `null` | S-series cores; omitted means `[0]`; X5 rejects this option |
| `--list-models` | flag | false | List manifest assets without SDK |
| `--dry-run` | flag | false | Resolve pair and tensor contract without SDK |

<a id="results"></a>
## Results

The CLI prints target, encoder/decoder asset IDs, input path, output mask path, selected mask index and IoU as JSON. It writes an overlay and a `512x512` binary mask. The result mask and low-resolution masks are owned copies of the pipeline result.

<a id="integration-example"></a>
## Integration Example

The complete example below assumes the S100 pair is prepared and runs on a board with `hbm_runtime`:

```python
import importlib
from pathlib import Path
import cv2

root = Path.cwd()
binding = importlib.import_module("samples.vision.mobile_sam.runtime.python.model_binding")
runner_type = importlib.import_module("samples.vision.mobile_sam.runtime.python.model_runner").RuntimeModelRunner
pipeline_type = importlib.import_module("samples.vision.mobile_sam.runtime.python.pipeline").MobileSAMPipeline
selection = binding.resolve_selection("s100")
runner = runner_type(selection)
bound = runner.load()
runner.set_scheduling_params(priority=0, bpu_cores=[0])
pipeline = pipeline_type(runner, bound)
image = cv2.imread(str(root / "samples/vision/mobile_sam/test_data/dogs.jpg"), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("test_data/dogs.jpg")
result = pipeline.predict(image, box=(185, 120, 380, 445))
assert result["mask"].shape == (512, 512)
```

<a id="stage-io"></a>
## Stage I/O

Each stage has its own three methods. Encoder `pre_process` validates BGR HWC input, applies the source ImageNet mean/std normalization and returns an owned contiguous RGB NCHW float32 tensor plus immutable per-call geometry context; encoder `forward` sends it and preserves the metadata-validated native embedding; encoder `post_process` owns a float32 copy. Decoder `pre_process` owns the embedding and box context; decoder `forward` sends the decoder tensors and preserves native `low_res_masks`/`iou_predictions`; decoder `post_process` casts accepted native numeric arrays to owned float32, selects the greatest IoU, resizes raw logits to `512x512`, and thresholds at `>0`. `predict` composes `encoder.pre_process → encoder.forward → encoder.post_process → decoder.pre_process → decoder.forward → decoder.post_process`. No implicit dequantization occurs. The decoder box follows actual metadata: S uses `[1,4]`, while X5 accepts `[1,4]` or `[1,4,1,1]`; masks are `[1,3,H,W]` with positive observed `H/W`, and IoU is `[1,3]` or `[1,3,1,1]`.

| Bound output | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Encoder embedding | `[1,256,32,32]` | `[1,256,32,32]` |
| Decoder masks | `[1,3,128,128]` | `[1,3,H,W]`; positive H/W read from actual metadata |
| Decoder IoU | `[1,3,1,1]` | `[1,3]` or `[1,3,1,1]`, as observed |

Accepted native output dtypes are `float16`, `float32`, `int8`, `uint8`, `int16`, `int32`; each array must exactly match its observed metadata. Their presence in this compatibility rule is not evidence of the published models' actual native dtype. The source float32 cast is preserved without dequantization. IoU is the model's predicted mask-quality score, not a ground-truth dataset measurement. Result fields are `mask` (owned bool `[512,512]`), `iou` (float), `mask_index` (integer 0–2), and `low_res_masks` (owned float32 `[1,3,H,W]`). PNG files encode false/true as 0/255. Instances are not advertised as SDK-thread-safe.

<a id="troubleshooting"></a>
## Troubleshooting

- Missing model: run the explicit model preparation command and check both paths.
- Custom path rejected: provide the exact matching stage asset ID.
- X5 core error: omit `--bpu-cores`; X5 has no selectable BPU core in this interface.
- Invalid box: pass four finite ordered coordinates within `[0,512]`.
- Shape mismatch: inspect `--dry-run` and runtime metadata; do not infer dimensions from a filename.
- Board identity or SDK error: verify the selected target and matching `hbm_runtime` image.
