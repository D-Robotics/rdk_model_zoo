# ResNet evaluation (ResNet18/50/152)

Use the bundled image for a functional board check. For dataset accuracy, prepare the matching labeled validation split and aggregate predictions using the metric definitions below. Reference tables list published measurements and their operating conditions.

<a id="dataset"></a>

## Dataset

Functional checks run on the bundled test images. For dataset-level
evaluation, prepare ImageNet validation data (50,000 images, ILSVRC2012
val) following [datasets/imagenet](../../../../datasets/imagenet/README.md);
the dataset is user-supplied (no download script ships here).

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="environment"></a>
## Environment

Host checks need the repository's user-space Python dependencies
(`requirements-host.txt` at the sample root) and no board SDK. The
functional board check needs the target board with its `hbm_runtime`
image, a prepared artifact, and the label file. The dataset-level
evaluation additionally needs the OE/board toolchain stated with the
result.

<a id="command"></a>
## Command

Host checks (cwd: repository root; success: all tests OK, exit 0):

```bash
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

Functional board check — X5 (prerequisite:
`bash samples/vision/resnet/model/download.sh x5 resnet18`; success: exit 0
and the expected Top-K):

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

Functional board check — S100 (S600 uses `s600` in the asset reference and
artifact path):

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target s100 \
  --asset-id s:resnet18:s100/resnet18_224x224_nv12.hbm \
  --model-path samples/vision/resnet/model/s100/resnet18_224x224_nv12.hbm \
  --test-img samples/vision/resnet/test_data/zebra_cls.jpg \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

ResNet50/152 (S100/S600) follow the same command with their
`s:resnet50|resnet152:<target>/...` references, matching artifact paths and
`--variant resnet50|resnet152`. The S-series C++ check runs
`bash samples/vision/resnet/runtime/cpp/run.sh`.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | any prepared artifact on its matching board |
| Top-K output | class IDs, scores and labels printed for the bundled image; repeated runs return the same list | same board, artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions | run on the prepared ImageNet val set; record the split with the result |
| latency / FPS | inference timing | measure on the target board; record batch/thread settings with the result (see the reference conditions below) |

<a id="outputs"></a>
## Outputs

Host checks print the unittest result. The functional board check prints
the Top-K (class ids, scores, labels) on stdout and optionally writes an
annotated image with `--img-save-path`. For evidence, save the board
identity, model reference, runtime metadata, raw F32 score tensor, Top-K
output, image path, resize type, and command line.

<a id="reference-results"></a>
## Reference results

Published X5 release record for ResNet18 (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS | Threading conditions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 224x224 | 1000 | 11.2 | 71.5% | 70.5% | 2.95 | 449+ | — (not stated by the source) |

ImageNet-val accuracy and latency are measured by preparing the dataset
above and recording the toolchain, batch and thread settings with the
result. The S release published no accuracy or latency figures for
ResNet18/50/152.

<a id="boundaries"></a>
## Boundaries

The checked-in material covers host contract tests and the functional
board check. Run dataset-level accuracy and latency through your own
harness on the prepared ImageNet val data, and record the board identity,
artifact, toolchain, batch and thread settings with every result. When a
board is unreachable or an artifact is unavailable, record that item
explicitly instead of dropping it.
