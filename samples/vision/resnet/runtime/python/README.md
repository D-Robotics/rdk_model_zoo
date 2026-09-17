# Python runtime

`main.py` is the canonical user-facing command. It resolves one exact model
reference from the release manifests, checks the detected board, loads
`hbm_runtime` lazily, and executes one `ClassificationTask` flow. Model
preparation is explicit; this runtime never downloads or installs packages.

## Prepare and run

Use the board Python environment that already contains the matching
`hbm_runtime`, NumPy, and OpenCV-Python. From the repository root, prepare the
artifact first:

```bash
bash samples/vision/resnet/model/download.sh x5
```

Then run it on the matching X5 board:

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names
```

For S100 or S600, use the matching `s:resnet18:<target>/...` reference and
artifact directory. `--list-models` prints all published references;
`--dry-run` resolves one without board access, model loading, or download.

## Command parameters

| Option | Behavior |
| --- | --- |
| `--target` | `auto`, `x5`, `s100`, `s100p`, or `s600`; an execution target must match detected hardware |
| `--asset-id` | complete `group:sample:filename` reference from the manifest |
| `--model-path` | existing `.bin` or `.hbm`; it must be paired with `--asset-id` |
| `--test-img` / `--label-file` | BGR input image and ImageNet labels |
| `--top-k` / `--topk` | number of results, default 5 |
| `--resize-type` | 0 direct stretch, or 1 letterbox with BGR 127 padding |
| `--priority` / `--bpu-cores` | runtime scheduling, default 0 and `[0]` |
| `--img-save-path` | optional annotated image |
| `--list-models` / `--dry-run` | SDK-free inspection modes |

The canonical API accepts one BGR `uint8` image:

```python
from samples.vision.resnet.runtime.python.classification import ClassificationTask
from samples.vision.resnet.runtime.python.model_binding import bind_model, resolve_selection
from samples.vision.resnet.runtime.python.model_runner import RuntimeModelRunner

selection = resolve_selection(
    "x5",
    asset_id="x5:resnet:resnet18_224x224_nv12.bin",
    model_path="samples/vision/resnet/model/resnet18_224x224_nv12.bin",
)
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = ClassificationTask(runner, binding, top_k=5)
result = task.predict(image)
print(result.class_ids, result.scores, result.labels)
```

The task returns `ClassificationResult(class_ids, scores, labels)`. X5 input is
packed as `(1,336,224,1)` uint8 NV12. S100/S600 input remains two arrays:
Y `(1,224,224,1)` and UV `(1,112,112,2)`. The tensor names, output name, and
F32 shape are validated against the selected contract before inference.

The source wrappers apply softmax to one returned score vector. The canonical
implementation keeps that behavior as `legacy_softmax` because available X5
conversion material does not establish whether its `prob` tensor is normalized
inside the graph or by the wrapper. It does not silently reinterpret output
semantics.

## Compatibility adapters

The former imports remain thin adapters:

```python
from platforms.x5.samples.vision.resnet.runtime.python.resnet import ResNet, ResNetConfig
from platforms.s.samples.vision.resnet18.runtime.python.resnet18 import Resnet18, Resnet18Config
```

The X5 adapter retains nested `{model_name: {input_name: tensor}}` inputs and
returns `(topk_idx, topk_prob, topk_labels)`. The S18 adapter retains nested
Y/UV inputs and returns a list of `(class_id, probability)` pairs. Their
`pre_process`, `forward`, and `post_process` methods delegate to
`tensor_io.prepare_nv12`, `RuntimeModelRunner`, and
`classification.topk_from_scores`. Passing `runtime=` or `runtime_factory=`
is an optional host fixture seam; normal board use constructs the installed
SDK lazily.

## Code flow and troubleshooting

```text
main.py
  -> model_binding.resolve_selection
  -> platforms.require_execution_target
  -> model_runner.RuntimeModelRunner.load
  -> model_binding.bind_model
  -> classification.ClassificationTask.predict
       -> tensor_io.prepare_nv12
       -> runtime.run
       -> classification.topk_from_scores
```

`model_path requires --asset-id` means the runtime cannot safely select an
input protocol from the filename. `Cannot identify this board` means automatic
identity detection is unavailable; an explicit target helps dry-run but does
not establish hardware evidence. Shape or dtype errors mean the artifact and
runtime metadata do not match the selected X5 packed or S-series split
contract. For output differences, compare raw F32 output, image, resize mode,
Top-K, and artifact before changing score processing.

Run the host contract and adapter tests with:

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```
