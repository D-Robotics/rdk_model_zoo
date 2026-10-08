# Python runtime utilities

[简体中文](README_cn.md)

Model Zoo samples share these functions for board SDK execution, image and tensor processing, labels, and visualization. Each sample defines its model-specific preprocessing and postprocessing.

## Directory structure

| Module | Purpose |
| --- | --- |
| `runtime.py` | Load a board model and call `hbm_runtime` |
| `model_runner.py`, `single_array_runner.py` | Bind classification and named-array model inputs/outputs |
| `platforms.py`, `platform_profile.py` | Resolve the board target and platform capabilities |
| `assets.py`, `cls_binding.py`, `sam_binding.py` | Read model manifests and tensor contracts |
| `image.py`, `file_io.py`, `preprocess.py`, `tensor_io.py` | Read images/files, resize and convert input tensors |
| `labels.py`, `classification.py` | Load and validate labels; construct classification results |
| `runtime_meta.py`, `quantization.py`, `nn_math.py`, `postprocess.py` | Interpret metadata, dequantize and process model outputs |
| `sam_runner.py`, `sam_stages.py`, `sam_tensor_io.py`, `sam_evaluator.py` | SAM encoder/decoder execution and evaluation |
| `yoloe26_decode.py`, `yoloe26_geometry.py` | YOLOE26 prompt-free decoding and coordinate restoration |
| `visualize.py`, `inspect.py` | Render results and inspect tensors |
| `text_metrics.py` | Compute transcript edit counts and character error rate |
| `tests/` | Unit tests |

## Usage

Run Python from the repository root, using the dependencies listed in the chosen sample's runtime guide.

```python
from pathlib import Path
from utils.py_utils.image import read_bgr_image, bgr_to_nv12_planes
from utils.py_utils.labels import load_labels, validate_labels

image = read_bgr_image("samples/vision/resnet/test_data/zebra_cls.jpg")
labels = load_labels(Path("datasets/imagenet/imagenet_classes.names"))
validate_labels(labels, 1000)
```

`read_bgr_image` returns a BGR `uint8` array `(H, W, 3)`. After resizing to the model's even-valued input dimensions, `bgr_to_nv12_planes` returns contiguous `uint8` Y `(1,H,W,1)` and UV `(1,H/2,W/2,2)` arrays. `validate_labels` accepts a complete sequence or an index/name mapping.

### Runtime

`RuntimeSession(model_path, target=...)` creates a session. Call `load()` on the matching board, then `run(inputs)` with the SDK's named input mapping. Model files and dependencies are prepared using the sample's model and runtime guides.

For a local classification model, use `RuntimeModelRunner.from_file(model_path, target=..., input_size=(height, width), class_count=...)`; call `load()` before using model metadata. Published model selections use `RuntimeModelRunner(selection, table=BINDING_TABLE)`, where the sample supplies its binding table.

See [ResNet](../../samples/vision/resnet/runtime/python/README.md) and [YOLO](../../samples/vision/ultralytics_yolo/runtime/python/README.md) for complete model APIs.

### Model selection and downloads

`assets.py` reads `docs/release/{x5,s}/models.yaml`. Asset references combine a group, sample and file identifier, for example `x5:resnet:resnet18_224x224_nv12.bin`. Use the sample's download command to prepare the selected artifact. Platform identities and aliases are defined in `docs/release/platforms.json`.

### Task helpers

SAM helpers bind encoder embeddings and decoder prompts. YOLOE26 helpers accept the ten NHWC float32 tensors of its 4585-class prompt-free protocol and restore boxes and masks using per-image geometry. See the [YOLOE conversion guide](../../samples/vision/yoloe/conversion/README.md) to prepare floating outputs.

`text_metrics.py` computes Unicode character error rate from saved transcripts, retaining whitespace, case, and punctuation. The [ASR evaluator](../../samples/speech/asr/evaluator/README.md) defines the input schema.

For function signatures and tensor descriptions, see the [source reference](../../docs/source_reference/README.md).
