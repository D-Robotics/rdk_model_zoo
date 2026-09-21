# EdgeNeXt evaluation

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
python3 -m unittest discover -s samples/vision/edgenext/tests -v
```

Functional board check on X5 (prerequisite:
`bash samples/vision/edgenext/model/download.sh x5 base`; success:
exit 0 and a Top-5 containing a zebra-related class):

```bash
python3 samples/vision/edgenext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin \
  --model-path samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin \
  --test-img samples/vision/edgenext/test_data/zebra.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

For a same-board before/after comparison, run the legacy platform
entrypoint (`platforms/x5/samples/vision/edgenext/runtime/python/main.py`)
with the same image, model bytes, labels, resize type, and Top-K, and
compare class IDs and Top-K scores before label formatting (identical IDs; scores judged within the tolerance planned for the B3 smoke — abs diff < 1e-5, following the executed B2 precedent recorded in the B2 board evidence; exact ties to be adjudicated with top-8 per-ID evidence; raw-tensor equality is not asserted). The output
should be finite, non-zero, and stable across repeated runs with the same
input.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | any prepared artifact on its matching board |
| Top-K agreement | identical post-softmax Top-K class IDs between canonical and legacy runs; scores within tolerance (bar: abs diff < 1e-5; exact ties adjudicated with recorded top-8 per-ID evidence) | same board, same artifact bytes, image, resize type, Top-K |
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
| board comparison (canonical vs legacy) | not-run (B3 board smoke pending; updated when executed) | — |
| dataset accuracy / latency | not-run in this sample | — |

Published historical figures from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3; source notes: Float Top-1 on the pre-quantization ONNX, Quant
Top-1 on the deployment model, latency single-frame single-thread
single-core, FPS multi-threaded; CPU 8xA55@1.8GHz performance mode, BPU
1xBayes-e@1GHz):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| EdgeNeXt-base | 224x224 | 18.51 | 78.21% | 74.52% | 8.80 | 113.35 |
| EdgeNeXt-small | 224x224 | 5.59 | 76.50% | 71.75% | 4.41 | 226.15 |
| EdgeNeXt-x-small | 224x224 | 2.34 | 71.75% | 66.25% | 2.88 | 345.73 |
| EdgeNeXt-xx-small | 224x224 | 1.33 | 69.50% | 64.25% | 2.47 | 403.49 |

The quantized Top-1 drops further below float for the smaller variants in
the published record; this is recorded as published, not re-measured or
explained here.

<a id="boundaries"></a>
## Boundaries

No dataset-level accuracy or latency harness ships with this sample: the
checked-in material covers host contract tests and functional board
checks only. Host test success never certifies a board. A board that is
unreachable or an artifact that is unavailable makes the corresponding
item `not-run`, not failed-and-forgotten.
