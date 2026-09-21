# EfficientViT evaluation

Evaluation has two separate purposes: confirm that one board executes the
selected artifact with the expected tensor contract, and measure accuracy
or latency with a stated dataset and toolchain. This directory documents
both; it contains no accuracy harness of its own (see
[boundaries](#boundaries)).

<a id="dataset"></a>
## Dataset

Not applicable for the current scope: this sample performs functional
checks (bundled test images) and does not run a dataset-level accuracy
evaluation. A dataset-based evaluation would require ImageNet validation
data (ILSVRC2012 val, 50,000 images) prepared separately by the user; no
dataset download or preparation script is provided.

<a id="environment"></a>
## Environment

Host checks need the repository's user-space Python dependencies
(`requirements-host.txt` at the sample root) and no board SDK. The
functional board check needs an X5 board with its `hbm_runtime` image, a
prepared artifact, and the label file. The dataset-level evaluation would
additionally need the OE/board toolchain stated with the result.

<a id="command"></a>
## Command

Host checks (cwd: repository root; success: all tests OK, exit 0):

```bash
python3 -m unittest discover -s samples/vision/efficientvit/tests -v
```

Functional board check on X5 (prerequisite:
`bash samples/vision/efficientvit/model/download.sh x5`; success: exit 0
and a Top-5 containing hook-related classes):

```bash
python3 samples/vision/efficientvit/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin \
  --model-path samples/vision/efficientvit/model/EfficientViT_m5_224x224_nv12.bin \
  --test-img samples/vision/efficientvit/test_data/hook.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

For a same-board before/after comparison, run the legacy platform
entrypoint (`platforms/x5/samples/vision/efficientvit/runtime/python/main.py`)
with the same image, model bytes, labels, resize type, and Top-K, and
compare class IDs and raw scores before label formatting. The output
should be finite, non-zero, and stable across repeated runs with the same
input.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | any prepared artifact on its matching board |
| Top-K agreement | identical class IDs and raw scores between canonical and legacy runs | same board, same artifact bytes, image, resize type, Top-K |
| Top-1 accuracy | fraction of argmax-correct predictions | ImageNet val — not evaluated in this sample |
| latency / FPS | inference timing | not evaluated in this sample; historical figures below carry unstated conditions |

<a id="outputs"></a>
## Outputs

Host checks print the unittest result. The functional board check prints
the Top-K (class ids, scores, labels) on stdout and optionally writes an
annotated image with `--img-save-path`. For evidence, save the board
identity, model reference, runtime metadata, raw F32 score tensor, Top-K
output, image path, resize type, and command line.

<a id="reference-results"></a>
## Reference results

| Item | Value | Source |
| --- | --- | --- |
| host tests | 26 OK (2026-09-21, author self-check) | migration evidence |
| board comparison (canonical vs legacy) | not-run (B2 board smoke pending; updated when executed) | — |
| dataset accuracy / latency | not-run in this sample | — |

Published historical figures from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3; source notes: Float Top-1 on the pre-quantization ONNX, Quant
Top-1 on the deployment model; the source table does not state the latency
threading conditions):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 | 12.4 | 73.75% | 72.50% | 6.34 | 174.70 |

The quantized Top-1 (72.50%) stays close to the float value (73.75%) in
the published record; this is recorded as published, not re-measured here.

<a id="boundaries"></a>
## Boundaries

No dataset-level accuracy or latency harness ships with this sample: the
checked-in material covers host contract tests and functional board
checks only. Host test success never certifies a board. A board that is
unreachable or an artifact that is unavailable makes the corresponding
item `not-run`, not failed-and-forgotten.
