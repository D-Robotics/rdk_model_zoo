# ResNet Python runtime

[`main.py`](main.py) provides the command line: construct the classifier,
call `predict()`, and display the results. [`classify.py`](classify.py) contains
`ResNetClassifier` initialization, preprocessing, inference, and postprocessing.
[`cli.py`](cli.py) groups command options, published model selection, and presentation.
`utils/py_utils/` provides image reading, label validation, and Runtime loading.

<a id="overview"></a>
## Python inference

Use this directory for python inference.

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

The target board requires matching `hbm_runtime`, NumPy, OpenCV-Python, and PyYAML.

| Board | Published models | Input |
| --- | --- | --- |
| RDK X5 | ResNet18 | packed NV12 |
| RDK S100 | ResNet18, ResNet50, ResNet152 | split NV12 |
| RDK S600 | ResNet18, ResNet50, ResNet152 | split NV12 |

See [model files](../../model/README.md) for artifact preparation and
[model conversion](../../conversion/README.md) for ONNX export and compilation.

<a id="usage"></a>
## Usage

Run from the repository root. List models and inspect a target configuration:

```bash
python3 samples/vision/resnet/runtime/python/main.py --list-models --target auto
python3 samples/vision/resnet/runtime/python/main.py --dry-run --target x5
```

Listing and dry-run work on a development host without loading the board SDK.
After preparing the model, run on the corresponding board:

```bash
python3 samples/vision/resnet/runtime/python/main.py --target x5
python3 samples/vision/resnet/runtime/python/main.py --target s100 --variant resnet50
python3 samples/vision/resnet/runtime/python/main.py --target s600 --variant resnet152
```

`--target auto` selects the board from local hardware; the default model is
ResNet18. An explicit target must match the executing board. To specify the
model path, image, and labels:

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--variant` | choice | null | model variant (`resnet18` on x5/s100/s600; `resnet50`/`resnet152` on s100/s600 only) |
| `--model-path` | string | null | existing `.bin`/`.hbm`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/resnet/test_data/white_wolf.JPEG | BGR input image |
| `--label-file` | string | null | one-label-per-line labels file; default: the bundled ImageNet labels |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | alias of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch or `1` letterbox with BGR 127 padding; default follows the model configuration |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--img-save-path` | string | null | optional annotated output image path |
| `--list-models` | flag | false | list manifest-backed references without board access |
| `--dry-run` | flag | false | resolve/check a selection without loading a model or SDK |

<a id="results"></a>
## Results

The command prints Top-K class IDs, labels, and scores. Add
`--img-save-path result.jpg` to save an annotated image. The library returns
`ClassificationResult(class_ids, scores, labels)`.

Published models use 224×224 inputs with letterbox resizing and BGR 127 padding
by default. X5 receives a flat uint8 NV12 array of 75,264 bytes; S100/S600 receive
Y `(1,224,224,1)` and UV `(1,112,112,2)` uint8 arrays. Published model outputs are
F32 score vectors; softmax is applied before stable descending Top-K selection.

<a id="integration-example"></a>
## Integration example

Import and reuse the classifier on the target board:

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier

model = ResNetClassifier(
    "samples/vision/resnet/model/resnet18_224x224_nv12.bin",
    target="x5", top_k=5,
)
result = model.predict("samples/vision/resnet/test_data/white_wolf.JPEG")
print(result.class_ids, result.scores, result.labels)
```

The CLI selects ResNet50 with `--target s100 --variant resnet50`; the library
accepts its `.hbm` path and `target="s100"`. `predict` accepts
an image path or a BGR uint8 NumPy array. Supply class names through `labels`;
without them, `result.labels` contains class ID strings. Input arrays remain unchanged.

<a id="custom-model"></a>
## Self-trained models

Declare the compiled artifact's board, input dimensions, and class count:

```python
from samples.vision.resnet.runtime.python.classify import ResNetClassifier

model = ResNetClassifier(
    "mymodels/my_resnet_4class.bin", target="x5",
    input_size=(224, 224), class_count=4, top_k=2,
    labels=["cat", "dog", "bus", "ship"],
)
result = model.predict("samples/vision/resnet/test_data/white_wolf.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`ResNetClassifier` defaults to F32 output with softmax. For outputs that are
already probabilities, set `score_policy="none"`; for quantized outputs,
set the corresponding `output_transform="dequant"`. The runtime checks actual
tensor shapes, types, and class count against the declaration. A label sequence
must contain one entry per class.

<a id="stage-io"></a>
## Stage interfaces

| Method | Input | Output |
| --- | --- | --- |
| `preprocess` | Image path or BGR uint8 array | `PreparedInput`: NV12 tensors and per-image resize information |
| `infer` | `PreparedInput` | Raw output tensor dictionary |
| `postprocess` | Raw output tensor dictionary | Top-K `ClassificationResult` |
| `predict` | Image path or BGR uint8 array | Runs the three stages and returns `ClassificationResult` |

`infer` calls `hbm_runtime` through the shared `RuntimeModelRunner`; the shared
`RuntimeSession` loads the model and checks board identity. Applications can call
the three stages separately to access intermediate data.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Action |
| --- | --- |
| `Cannot identify this board` | Run inference on a supported board; inspect configuration on a host with `--dry-run --target x5`. |
| `model file not found` | Prepare the artifact from the model files guide, or specify `--asset-id` and `--model-path`. |
| `model_path requires --asset-id` | Copy the full reference from `--list-models`. |
| No published model for S100P | Published models support X5, S100, and S600; declare a self-trained S100P model with `ResNetClassifier`. |
| Input/output shape or type mismatch | Check the compilation target, input protocol, output class count, and selected model configuration. |
