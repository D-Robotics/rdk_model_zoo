# EfficientNet Python runtime

`main.py` is the canonical user-facing command. It resolves one exact model
reference from the release manifests, checks the detected board, loads
`hbm_runtime` lazily, and executes one `ClassificationTask` flow. Model
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

cwd: repository root. The minimal SDK-free invocation is the listing mode:

```bash
# success: prints all 13 published references, exit 0, no SDK loaded
python3 samples/vision/efficientnet/runtime/python/main.py --list-models --target auto
```

A full run on a prepared X5 board:

```bash
# prerequisites: bash samples/vision/efficientnet/model/download.sh x5 b2
# success: exit 0 and a printed Top-5 list
python3 samples/vision/efficientnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientnet:EfficientNet_B2_224x224_nv12.bin \
  --model-path samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin \
  --test-img samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

An S100 run with a non-224 variant (the geometry follows the variant —
lite1 is 240x240):

```bash
# prerequisites: bash samples/vision/efficientnet/model/download.sh s100 lite1
python3 samples/vision/efficientnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm \
  --model-path samples/vision/efficientnet/model/s100/efficientnet_lite1_240x240_nv12.hbm
```

S600 substitutes the `s600/` reference and path. `--dry-run --target x5`
resolves a selection without board access, model loading, or download;
`run.sh` in this directory forwards its arguments to `main.py` unchanged.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--variant` | choice | null | model variant (`b2` default when omitted; see `--list-models` for published combinations) |
| `--model-path` | string | null | existing `.bin`/`.hbm`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG | BGR input image |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | one-label-per-line ImageNet labels |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | legacy spelling of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch or `1` letterbox with BGR 127 padding; default follows the bound source |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--img-save-path` | string | null | optional annotated output image path |
| `--list-models` | flag | false | list manifest-backed references without board access |
| `--dry-run` | flag | false | resolve/check a selection without loading a model or SDK |

Defaults above are machine-checked against `build_parser()` by the Q3
checker.

<a id="results"></a>
## Results

The command prints the stable Top-K as class IDs, scores, and labels
(`ClassificationResult(class_ids, scores, labels)`), and writes an
annotated image only when `--img-save-path` is given. X5 receives the packed
NV12 buffer as the canonical flat 1-D uint8 array of `H*W*3/2` bytes
(224x224 -> 75,264 bytes); S100/S600 receive Y `(1,H,W,1)` and UV
`(1,H/2,W/2,2)` uint8 arrays where H=W follows the variant — lite0 224 ->
Y `(1,224,224,1)`, UV `(1,112,112,2)`; lite4 380 -> Y `(1,380,380,1)`, UV
`(1,190,190,2)`. The published artifacts return raw logits; the task
applies a stable softmax before Top-K on both platforms. Default resize is
letterbox (type 1) on both platforms; the interpolation differs by source —
X5 resizes with linear interpolation, S with nearest-neighbor.
Output shapes follow the rank rule: any singleton-batch/spatial spelling
that squeezes to `(1000,)` binds (the published artifacts declare the
`raw_f32` transform; quantized artifacts would require a declared `dequant`
contract).

<a id="integration-example"></a>
## Integration example

Prerequisites: artifact prepared (see [model/README.md](../../model/README.md))
and OpenCV-Python importable. Every input variable is defined in the example:

```python
import cv2

from samples.vision.efficientnet.runtime.python.classification import ClassificationTask
from samples.vision.efficientnet.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.efficientnet.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection(
    "s100",
    asset_id="s:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm",
    model_path="samples/vision/efficientnet/model/s100/efficientnet_lite1_240x240_nv12.hbm",
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = ClassificationTask(runner, binding, top_k=5)
image = cv2.imread("samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG")
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

The three stages can also be driven explicitly: `prepared = task.pre_process(image)`,
`outputs = task.forward(prepared.tensors)`,
`result = task.post_process(outputs)` — `predict` chains exactly these
stages (verified by the stage-contract tests).

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output |
| --- | --- | --- |
| `pre_process` | one BGR `uint8` array (any size) | `PreparedInput.tensors` (target-shaped NV12 tensors at the variant geometry) + `PreparedInput.transform` (frozen per-call resize context) |
| `forward` | `prepared.tensors` | raw output dict (X5 F32 `[1,1000,1,1]`; S F32 `[1,1000]`) — bit-identical to the runner output, no decode |
| `post_process` | raw outputs (no context: classification consumes no geometry) | `ClassificationResult(class_ids, scores, labels)`, stable descending Top-K under the declared score policy |
| `predict` | BGR `uint8` array | chains the three stages, same `ClassificationResult` |

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Cannot identify this board` | Run with an explicit target for dry-run, then execute only on that matching board; an explicit target is not hardware evidence. |
| `model_path requires --asset-id` | Copy the exact qualified reference from `--list-models`; do not use a bare filename. |
| `No published ... asset` for S100P | There is no s100p asset row in the manifest; use S100/S600 artifacts on their matching boards (the legacy S wrapper silently fell back to the lite0 S100 build — this sample errors instead). |
| input shape or dtype mismatch | Confirm the artifact reference and runtime metadata; a lite1 artifact declares 240x240 and rejects 224 metadata rather than resizing in secret. |
| output differs from a legacy run | Compare the same artifact, image, resize mode, Top-K, and raw output before changing score semantics. |

Host checks (repository root):
`python3 -m unittest discover -s samples/vision/efficientnet/tests -v`.
