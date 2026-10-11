English | [简体中文](README_cn.md)

# MobileNetV4 Python runtime

<a id="overview"></a>
## Python inference

[`main.py`](main.py) parses arguments, constructs `MobileNetV4Classifier`, calls `predict`, and displays results.
[`classify.py`](classify.py) contains model initialization, preprocessing, inference, and postprocessing.
[`cli.py`](cli.py) groups command options, published model selection, and result presentation.
Image reading, label validation, and SDK sessions use `utils/py_utils/`.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── classify.py  # Classification preprocessing, inference, and postprocessing
├── cli.py  # Arguments and result presentation
├── main.py  # Command-line entry
└── run.sh  # Run the sample
```

<a id="environment"></a>
## Environment

Run on the target board's Python environment with the matching
`hbm_runtime`, NumPy, OpenCV-Python, and Pillow; PyYAML is needed for manifest
reading. Pillow performs the antialiased bicubic resize of the shorter-edge
center crop the published models expect (install it with
`python3 -m pip install pillow` if the board image lacks it).
`hbm_runtime` is supplied by the board image and is imported lazily.
`--help`, `--list-models`, and `--dry-run` inspect the selection without
executing a model. Host-side test dependencies are listed in the sample's
`requirements-host.txt`.

<a id="usage"></a>
## Usage

cwd: repository root. Default command (zero arguments beyond the mode
flags is not executable without a board and artifact, so the minimal
SDK-free invocation is the listing mode):

```bash
# success: prints all published references, exit 0, no SDK loaded
python3 samples/vision/mobilenetv4/runtime/python/main.py --list-models --target auto
```

A full run on a prepared X5 board:

```bash
# prerequisites: bash samples/vision/mobilenetv4/model/download.sh x5 small
# success: exit 0 and a printed Top-5 list
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100, S100P and S600 substitute the `s:` reference (for example
`s:mobilenetv4:s100p/mobilenetv4_conv_small_nashm_224x224_nv12.hbm`) and the
`s100/`, `s100p/` or `s600/` artifact path; the labels file is shared. `--dry-run --target x5` resolves a selection without board access,
model loading, or download.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--variant` | choice | null | model variant: `small`, `medium`, `large` (`small` default when omitted; see `--list-models` for published combinations) |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--model-path` | string | null | existing `.bin`/`.hbm`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG | BGR input image |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | one-label-per-line ImageNet labels |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | alias of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch, `1` letterbox with BGR 127 padding, or `2` shorter-edge resize + center crop (needs Pillow); default follows the bound model (`2`) |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--img-save-path` | string | null | optional annotated output image path |
| `--list-models` | flag | false | list manifest-backed references without board access |
| `--dry-run` | flag | false | resolve/check a selection without loading a model or SDK |

Defaults above are the values defined in `build_parser` ([cli.py](cli.py)).

<a id="results"></a>
## Results

The command prints the stable Top-K as class IDs, scores, and labels
(`ClassificationResult(class_ids, scores, labels)`), and writes an
annotated image only when `--img-save-path` is given. X5 receives the packed
NV12 buffer as the flat 1-D uint8 array of `H*W*3/2` bytes
(224x224 -> 75,264 bytes; 256x256 -> 98,304 bytes);
S100/S100P/S600 receive Y `(1,H,W,1)` and UV `(1,H/2,W/2,2)` uint8
arrays (Small and Medium are 224x224, Large is 256x256 on every target).
The published artifacts return raw logits; the task applies a stable
softmax before Top-K on both platforms.
Output shapes follow the rank rule: any singleton-batch/spatial spelling
that squeezes to `(1000,)` binds (the published artifacts declare the
`raw_f32` transform; quantized artifacts would require a declared `dequant`
contract).

<a id="integration-example"></a>
## Integration example

Prerequisites: artifact prepared (see [model/README.md](../../model/README.md)),
and OpenCV-Python and Pillow importable. Every input variable is defined in the example:

```python
from samples.vision.mobilenetv4.runtime.python.classify import MobileNetV4Classifier
from samples.vision.mobilenetv4.runtime.python.cli import resolve_selection

selection = resolve_selection(
    "x5",
    asset_id="x5:mobilenetv4:mobilenetv4_conv_small_bayese_224x224_nv12.bin",
    model_path="samples/vision/mobilenetv4/model/mobilenetv4_conv_small_bayese_224x224_nv12.bin",
)
contract = selection.contract
model = MobileNetV4Classifier(
    selection.model_path, target=selection.target,
    input_size=(contract.input_height, contract.input_width),
    class_count=contract.class_count, top_k=5,
    resize_type=contract.resize_type,
    resize_interpolation=contract.resize_interpolation,
    resize_shorter=contract.resize_shorter,
    score_policy=contract.output_score_policy,
    output_transform=contract.output_transform,
)
result = model.predict("samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`predict` accepts a local image path or BGR `uint8` NumPy array; the input array remains unchanged. The three stages can also be driven explicitly:
`prepared = model.preprocess(source)`, `outputs = model.infer(prepared)`,
`result = model.postprocess(outputs)` — `predict` chains exactly these
steps.

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output |
| --- | --- | --- |
| `preprocess` (`pre_process`) | image path or one BGR `uint8` array (any size) | `PreparedInput.tensors` (target-shaped NV12 tensors) + `PreparedInput.transform` (frozen per-call resize context, including the crop offsets) |
| `infer` (`forward`) | `PreparedInput` | raw output dict (X5 F32 `[1,1000,1,1]`; S F32 `[1,1000]`) — raw SDK tensors before score processing |
| `postprocess` (`post_process`) | raw outputs (no context: classification consumes no geometry) | `ClassificationResult(class_ids, scores, labels)`, stable descending Top-K under the declared score policy |
| `predict` | image path or BGR `uint8` array | chains the three stages, same `ClassificationResult` |

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Pillow is required for resize_type 2` | Install Pillow in the board's Python environment (`python3 -m pip install pillow`). |
| `Cannot identify this board` | Select a target supported by the sample and run inference on the matching board. |
| `model_path requires --asset-id` | Pass the exact qualified reference from `--list-models` with `--asset-id`. |
| input shape or dtype mismatch | Check that the artifact reference and target use the expected packed X5 or split S tensor layout. |
| output differs from a reference run | Compare the same artifact, image, resize mode, Top-K, and raw output before changing score semantics. |
