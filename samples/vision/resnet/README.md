# ResNet18 image classification

This sample runs the ResNet18 ImageNet classifier on an RDK X5, S100, or S600.
It has one maintained Python flow and one maintained S-series C++ flow. The
former X5 and S18 Python entrypoints remain usable compatibility shims and call
the same canonical implementation.

[中文说明](README_cn.md) · [Python runtime](runtime/python/README.md) ·
[model preparation](model/README.md) · [conversion](conversion/README.md) ·
[evaluation](evaluator/README.md)

## Choose the board and artifact

Use the row matching the board on which the command will run. These are exact
qualified references to existing rows in the platform release manifests; the
reference is also required when a custom local model path is supplied.

| Board | Reference | Local artifact produced by the downloader | Input passed to runtime |
| --- | --- | --- | --- |
| X5 | `x5:resnet:resnet18_224x224_nv12.bin` | `model/resnet18_224x224_nv12.bin` | one packed NV12 tensor |
| S100 | `s:resnet18:s100/resnet18_224x224_nv12.hbm` | `model/s100/resnet18_224x224_nv12.hbm` | separate Y and UV tensors |
| S600 | `s:resnet18:s600/resnet18_224x224_nv12.hbm` | `model/s600/resnet18_224x224_nv12.hbm` | separate Y and UV tensors |

The manifests are the authority for URL, format, and optional publisher hash.
There is no ResNet18 asset row for S100P, so selection rejects S100P instead of
guessing compatibility. ResNet50 and ResNet152 remain legacy material and are
outside this runnable ResNet18 sample.

## Prepare the environment

Board Python execution needs the board image's matching `hbm_runtime`, NumPy,
and OpenCV-Python. The package is imported only when a model is executed; help,
listing, dry-run, and host tests work without the board SDK. Do not install a
different SDK into the board image to work around a target mismatch.

The manifest reader also needs PyYAML. On a development host, use an isolated
environment and install the user-space dependencies from
`requirements-host.txt`:

```bash
python3 -m venv .venv-resnet
source .venv-resnet/bin/activate
python3 -m pip install -r samples/vision/resnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

The board's preinstalled `hbm_runtime` is supplied by the RDK image and is not
part of this host requirements file.

The C++ example is the source S18 implementation consolidated under this
sample. Its board build needs CMake, a C++17 compiler, OpenCV development
headers/libraries, `gflags`, and the installed Horizon DNN headers/libraries
(`hbDNN`, `hbucp`, and `fmt`). The build reads the board SoC identity from
`/sys/class/boardinfo/soc_name`; it is intended to run on an S100 or S600 board.
The launcher does not run `apt`, `pip`, or a model download.

Model conversion runs on an x86 Linux OpenExplore environment, not on the
board. See [conversion/README.md](conversion/README.md) for the source model,
the explicit ONNX export command, the OE sample path, and the artifacts that
must be recorded.

## Prepare the model explicitly

From the repository root, fetch exactly one published artifact into the
canonical model directory:

```bash
bash samples/vision/resnet/model/download.sh x5
bash samples/vision/resnet/model/download.sh s100
bash samples/vision/resnet/model/download.sh s600
```

The same operation is available as `python3
samples/vision/resnet/model/download.py --target s100`. The downloader reads
the selected row from `platforms/x5/docs/release/models.yaml` or
`platforms/s/docs/release/models.yaml`, writes through a temporary file, checks
the recorded hash when one exists, and finishes atomically. The current rows
have no publisher SHA-256, so it prints that the observed digest cannot prove
origin. Downloading is explicit and never occurs during inference.

If the artifact is stored elsewhere, pass its path together with the exact
reference from the table. A bare filename is not enough to select packed versus
split NV12.

## Run the Python sample

Use `--help` to inspect all options. These commands are complete examples after
the matching artifact has been prepared on the target board:

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names

python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file platforms/s/datasets/imagenet/imagenet_classes.names

python3 samples/vision/resnet/runtime/python/main.py \
  --target s600 \
  --asset-id s:resnet18:s600/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file platforms/s/datasets/imagenet/imagenet_classes.names
```

For a board-local model path, use `--target auto` only when the board identity
is readable. `--list-models` and `--dry-run` are useful before execution:

```bash
python3 samples/vision/resnet/runtime/python/main.py --list-models --target auto
python3 samples/vision/resnet/runtime/python/main.py --dry-run --target x5
python3 samples/vision/resnet/runtime/python/main.py --dry-run \
  --asset-id s:resnet18:s600/resnet18_224x224_nv12.hbm
```

The command prints stable Top-K class IDs, scores, and labels. `--top-k` and
the legacy spelling `--topk` default to 5. `--resize-type 0` stretches to
224x224; `--resize-type 1` preserves aspect ratio and pads with BGR 127, which
is the source default. X5 uses linear interpolation for direct resize; S100
and S600 retain the source nearest-neighbor direct resize.

## Run the consolidated C++ sample

After preparing an S100 or S600 artifact and the board C++ dependencies, run:

```bash
bash samples/vision/resnet/runtime/cpp/run.sh
```

The default launcher expects the S100 artifact under `model/s100`. For S600,
select the artifact explicitly and use a separate build directory if the same
checkout is used on both boards:

```bash
MODEL_PATH="$PWD/samples/vision/resnet/model/s600/resnet18_224x224_nv12.hbm" \
BUILD_DIR="$PWD/samples/vision/resnet/runtime/cpp/build-s600" \
bash samples/vision/resnet/runtime/cpp/run.sh
```

The launcher invokes CMake, builds `resnet18`, and passes the model, bundled
image, and S ImageNet label file to the binary. It checks those files first and
does not alter the system. The C++ implementation uses the existing S utility
sources from `platforms/s/utils/c_utils`; the numerical preprocess, BPU call,
and Top-K postprocess were kept as source code, not replaced by a link or a
second platform tree. X5 has no equivalent C++ source in the audited baseline.

## Library flow and output contract

The canonical Python flow is:

```text
resolve_selection -> require_execution_target -> RuntimeModelRunner.load
  -> bind_model -> ClassificationTask.predict
       -> tensor_io.prepare_nv12 -> runtime call
       -> classification.topk_from_scores
```

`ClassificationTask` accepts a BGR `uint8` array and returns
`ClassificationResult(class_ids, scores, labels)`. `RuntimeModelRunner` is the
only SDK boundary and validates runtime names, shapes, dtypes, and the single
F32 score output before returning it. X5 receives packed `(1,336,224,1)` uint8
NV12; S100/S600 receive Y `(1,224,224,1)` and UV `(1,112,112,2)` uint8 arrays.

Both audited Python wrappers apply softmax to the returned score vector. The
canonical contract records this as `legacy_softmax` over an
`unverified_score_vector`: the X5 tensor is named `prob` and appears normalized,
but the available conversion material does not establish whether normalization
belongs to the graph or wrapper. This sample keeps the source behavior and does
not claim a new output semantic.

The compatibility imports preserve the former APIs:

```python
from platforms.x5.samples.vision.resnet.runtime.python.resnet import ResNet, ResNetConfig
from platforms.s.samples.vision.resnet18.runtime.python.resnet18 import Resnet18, Resnet18Config
```

`ResNet.pre_process`/`forward`/`post_process` retain the X5 nested input and
`(topk_idx, topk_prob, topk_labels)` tuple. `Resnet18` retains nested split
inputs and a list of `(class_id, probability)` pairs. Both instantiate the
canonical binding and delegate preprocessing and decoding; optional
`runtime=`/`runtime_factory=` keywords are host-test seams.

## Source mapping and limits

| Audited source | Canonical symbol or file | Preserved local behavior |
| --- | --- | --- |
| X5 `ResNet.pre_process` | `ClassificationTask.pre_process` → `tensor_io.prepare_nv12` | packed NV12 and linear direct resize |
| S18 `Resnet18.pre_process` | `ClassificationTask.pre_process` → `tensor_io.prepare_nv12` | separate Y/UV tensors and nearest direct resize |
| X5/S18 `forward` | `RuntimeModelRunner.__call__` | nested legacy adapters, flat canonical runner |
| X5/S18 `post_process` | `classification.topk_from_scores` | legacy softmax and stable descending Top-K |
| X5/S18 Python `main.py` | canonical `runtime/python/main.py` | old paths supply defaults and exact manifest IDs |
| S18 `runtime/cpp/{inc,src}` | canonical `runtime/cpp/{inc,src}` | hbDNN calls, image utility calls, and output flow retained |
| X5/S18 model scripts | canonical `model/download.py` | manifest URL/format/hash ownership and atomic writes |

The old ResNet50/152 conversion and runtime files are not presented as
ResNet18 implementations. The audited ResNet18 source points to the official
OE classification example but does not ship its YAML, calibration set, or a
replayable board conversion recipe. The canonical exporter therefore creates a
fixed-shape ONNX source graph, while the exact weights and OE configuration
used for each published artifact still need to be recorded before claiming
artifact equivalence. See the conversion limitations in
[conversion/README.md](conversion/README.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `Cannot identify this board` | Run with an explicit target for dry-run, then execute only on that matching board; an explicit target is not hardware evidence. |
| `model_path requires --asset-id` | Copy the exact qualified reference from `--list-models`; do not use a bare filename. |
| `No published ... asset` for S100P | There is no ResNet18 S100P row in the release manifest. Use S100 or S600 artifacts only on their matching boards. |
| input shape or dtype mismatch | Confirm the artifact reference and runtime metadata; do not swap packed X5 and split S artifacts. |
| model download hash warning | The manifest row has no publisher SHA-256. Preserve the observed digest in local evidence and verify its source separately. |
| C++ configure/build failure | Confirm OpenCV/gflags/Horizon DNN development packages, compiler C++17 support, and the board SoC files; the launcher does not install them. |
| output differs from a legacy run | Compare the same artifact, image, resize mode, Top-K, and raw output before changing score semantics. |

Run the host checks from the repository root with:

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

Host checks validate selection, metadata rejection, image geometry, packed and
split tensor layouts, adapter API behavior, and numerical Top-K decoding. Board
execution and native C++ results must be recorded separately with the board
identity, artifact reference, command, and raw output.
