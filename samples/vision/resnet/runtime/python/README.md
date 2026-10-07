# ResNet18 Python runtime

`main.py` is the canonical user-facing command: it parses arguments,
constructs the model, calls `predict`, and shows the result. The complete
classification flow lives in [`classify.py`](classify.py):
`ResNetClassifier` shows initialization, `preprocess`, `infer`,
`postprocess` and `predict` in one readable file. It resolves one exact
model reference from the release manifests (or an explicit custom
contract), checks the detected board, loads `hbm_runtime` lazily through
the shared SDK session, and executes one classification flow. Model
preparation is explicit; this runtime never downloads or installs packages.

<a id="environment"></a>
## Environment

Run on the target board's Python environment with the matching
`hbm_runtime`, NumPy, and OpenCV-Python; PyYAML is needed for manifest
reading. `hbm_runtime` exists only in board images and is imported lazily —
`--help`, `--list-models`, `--dry-run`, and the host unittest suite run
without it. Host-side test dependencies are listed in the sample's
`requirements-host.txt`.

<a id="usage"></a>
## Usage

Run from the repository root. To inspect manifest-backed model references
without loading the board SDK, use listing mode:

```bash
# success: prints all published references, exit 0, no SDK loaded
python3 samples/vision/resnet/runtime/python/main.py --list-models --target auto
```

A full run on a prepared X5 board:

```bash
# prerequisites: bash samples/vision/resnet/model/download.sh x5
# success: exit 0 and a printed Top-5 list
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 substitute the `s:resnet18:<target>/...` reference and the
`s100/`/`s600/` artifact path; the labels file is shared.
`--dry-run --target x5` resolves a selection without board access, model
loading, or download.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--variant` | choice | null | model variant (`resnet18` on all targets; `resnet50`/`resnet152` on s100/s600 only) |
| `--model-path` | string | null | existing `.bin`/`.hbm`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/resnet/test_data/white_wolf.JPEG | BGR input image |
| `--label-file` | string | null | one-label-per-line labels file; default: the bundled ImageNet labels for 1000-class models, raw class IDs for custom class counts |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | legacy spelling of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch or `1` letterbox with BGR 127 padding; default follows the bound source |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--img-save-path` | string | null | optional annotated output image path |
| `--list-models` | flag | false | list manifest-backed references without board access |
| `--dry-run` | flag | false | resolve/check a selection without loading a model or SDK |

<a id="results"></a>
## Results

The command prints the stable Top-K as class IDs, scores, and labels
(`ClassificationResult(class_ids, scores, labels)`), and writes an
annotated image only when `--img-save-path` is given. X5 receives the packed
NV12 buffer as the canonical flat 1-D uint8 array of `H*W*3/2` bytes
(224x224 -> 75,264 bytes; same bytes as the former `(1,336,224,1)` view);
S100/S600 receive Y `(1,224,224,1)` and UV `(1,112,112,2)` uint8 arrays.
The classifier applies softmax to the returned score vector before selecting
the stable descending Top-K. Published artifacts declare `raw_f32` output;
custom quantized artifacts must declare the appropriate `dequant` transform
in their binding contract. Output shapes follow the rank rule: any
singleton-batch/spatial spelling that squeezes to `(1000,)` binds.

<a id="integration-example"></a>
## Integration example

Prerequisites: artifact prepared (see [model/README.md](../../model/README.md))
and OpenCV-Python importable. Every input variable is defined in the example:

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier
from samples.vision.resnet.runtime.python.model_binding import resolve_selection

selection = resolve_selection(
    "x5",
    asset_id="x5:resnet:resnet18_224x224_nv12.bin",
    model_path="samples/vision/resnet/model/resnet18_224x224_nv12.bin",
)
model = ResNetClassifier(selection, top_k=5)
result = model.predict("samples/vision/resnet/test_data/white_wolf.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`predict` accepts a local image path or a BGR `uint8` NumPy array and never
modifies the array in place. The three stages can also be driven explicitly:
`prepared = model.preprocess(image)`, `outputs = model.infer(prepared)`,
`result = model.postprocess(outputs)` — `predict` chains these three
steps. The established
`pre_process` / `forward` / `post_process` spellings remain thin aliases, and
the shared `ClassificationTask` flow stays importable from
[`classification.py`](classification.py).

<a id="custom-model"></a>
## Custom (self-trained) models

A locally compiled classifier needs no manifest registration. Declare the
contract the artifact was built for; binding still validates the actual
runtime tensor names, shapes, dtypes and class count against it:

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier
from samples.vision.resnet.runtime.python.model_binding import custom_selection

selection = custom_selection(
    "mymodels/my_resnet_4class.bin", "x5",
    input_height=224, input_width=224, class_count=4,
)
model = ResNetClassifier(selection, top_k=2, labels=["cat", "dog", "bus", "ship"])
image_path = "samples/vision/resnet/test_data/white_wolf.JPEG"  # replace with your image path
result = model.predict(image_path)
```

Without `labels`, results keep the raw class IDs — ImageNet names are never
assumed for a custom class count. A label list whose length differs from the
class count fails with a concrete error instead of mislabeling results.

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output |
| --- | --- | --- |
| `preprocess` (`pre_process`) | image path or one BGR `uint8` array (any size) | `PreparedInput.tensors` (target-shaped NV12 tensors) + `PreparedInput.transform` (frozen per-call resize context) |
| `infer` (`forward`) | `PreparedInput` | raw `{'prob': ndarray}` (X5, F32 `[1,1000,1,1]`) or `{'output': ndarray}` (S, F32 `[1,1000]`) — bit-identical to the runner output, no decode |
| `postprocess` (`post_process`) | raw outputs (no context: classification consumes no geometry) | `ClassificationResult(class_ids, scores, labels)`, stable descending Top-K after `legacy_softmax` |
| `predict` | image path or BGR `uint8` array | chains the three stages, same `ClassificationResult` |

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Cannot identify this board` | Set an explicit target for `--dry-run`; run inference on the matching board. |
| `model_path requires --asset-id` | Copy the exact qualified reference from `--list-models`; do not use a bare filename. |
| `No published... asset` for S100P | There is no ResNet18 S100P row in the manifest; use S100/S600 artifacts on their matching boards. |
| input shape or dtype mismatch | Confirm the artifact reference and runtime metadata; do not swap packed X5 and split S artifacts. |
| output differs from a legacy run | Compare the same artifact, image, resize mode, Top-K, and raw output before changing score semantics. |
