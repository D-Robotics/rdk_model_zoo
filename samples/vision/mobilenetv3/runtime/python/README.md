# MobileNetV3 Python runtime

`main.py` is the canonical user-facing command: it parses arguments,
constructs the model, calls `predict`, and shows the result. The complete
classification flow lives in [`classify.py`](classify.py):
`MobileNetV3Classifier` shows initialization, `preprocess`, `infer`, `postprocess` and
`predict` in one readable file. It resolves one exact model reference
from the release manifests, checks the detected board, loads
`hbm_runtime` lazily through the shared SDK session, and executes one
classification flow. Prepare the target artifact with the model downloader before running inference. The runtime resolves the selected manifest reference and loads the matching board SDK.

<a id="environment"></a>
## Environment

Run on the target board's Python environment with the matching
`hbm_runtime`, NumPy, and OpenCV-Python; PyYAML is needed for manifest
reading. `hbm_runtime` is supplied by the board image and is imported lazily.
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
python3 samples/vision/mobilenetv3/runtime/python/main.py --list-models --target auto
```

A full run on a prepared X5 board:

```bash
# prerequisites: bash samples/vision/mobilenetv3/model/download.sh x5
# success: exit 0 and a printed Top-5 list
python3 samples/vision/mobilenetv3/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv3:MobileNetV3_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv3/model/MobileNetV3_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv3/test_data/kit_fox.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 substitute the `s:` reference (for example
`s:mobilenetv3:s100/mobilenetv3_224x224_nv12.hbm`) and the `s100/`/`s600/` artifact path; the labels file is
shared. `--dry-run --target x5` resolves a selection without board access,
model loading, or download.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--model-path` | string | null | existing `.bin`/`.hbm`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/mobilenetv3/test_data/kit_fox.JPEG | BGR input image |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | one-label-per-line ImageNet labels |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | compatibility spelling of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch or `1` letterbox with BGR 127 padding; default follows the bound source |
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
NV12 buffer as the canonical flat 1-D uint8 array of `H*W*3/2` bytes
(224x224 -> 75,264 bytes; same bytes as the former `(1,336,224,1)` view);
S100/S600 receive Y `(1,224,224,1)` and UV `(1,112,112,2)` uint8
arrays.
The published artifacts return raw logits; the task applies a stable
softmax before Top-K on both platforms.
Output shapes follow the rank rule: any singleton-batch/spatial spelling
that squeezes to `(1000,)` binds (the published artifacts declare the
`raw_f32` transform; quantized artifacts would require a declared `dequant`
contract).

<a id="integration-example"></a>
## Integration example

Prerequisites: artifact prepared (see [model/README.md](../../model/README.md))
and OpenCV-Python importable. Every input variable is defined in the example:

```python
from samples.vision.mobilenetv3.runtime.python.classify import MobileNetV3Classifier
from samples.vision.mobilenetv3.runtime.python.model_binding import resolve_selection

selection = resolve_selection(
    "x5",
    asset_id="x5:mobilenetv3:MobileNetV3_224x224_nv12.bin",
    model_path="samples/vision/mobilenetv3/model/MobileNetV3_224x224_nv12.bin",
)
model = MobileNetV3Classifier(selection, top_k=5)
result = model.predict("samples/vision/mobilenetv3/test_data/kit_fox.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`predict` accepts a local image path or BGR `uint8` NumPy array; the input array remains unchanged. The three stages can also be driven explicitly:
`prepared = model.preprocess(source)`, `outputs = model.infer(prepared)`,
`result = model.postprocess(outputs)` — `predict` chains exactly these
steps. The shared `ClassificationTask` flow stays importable from
[`classification.py`](classification.py).

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output |
| --- | --- | --- |
| `preprocess` (`pre_process`) | image path or one BGR `uint8` array (any size) | `PreparedInput.tensors` (target-shaped NV12 tensors) + `PreparedInput.transform` (frozen per-call resize context) |
| `infer` (`forward`) | `PreparedInput` | raw output dict (X5 F32 `[1,1000,1,1]`; S F32 `[1,1000]`) — bit-identical to the runner output, no decode |
| `postprocess` (`post_process`) | raw outputs (no context: classification consumes no geometry) | `ClassificationResult(class_ids, scores, labels)`, stable descending Top-K under the declared score policy |
| `predict` | image path or BGR `uint8` array | chains the three stages, same `ClassificationResult` |

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Cannot identify this board` | Select a target supported by the sample and run inference on the matching board. |
| `model_path requires --asset-id` | Pass the exact qualified reference from `--list-models` with `--asset-id`. |
| input shape or dtype mismatch | Check that the artifact reference and target use the expected packed X5 or split S tensor layout. |
| output differs from a reference run | Compare the same artifact, image, resize mode, Top-K, and raw output before changing score semantics. |

Host checks (repository root):
`python3 -m unittest discover -s samples/vision/mobilenetv3/tests -v`.
