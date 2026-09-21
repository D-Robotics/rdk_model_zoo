# EfficientNet evaluation

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
functional board check needs the target board with its `hbm_runtime`
image, a prepared artifact, and the label file. The dataset-level
evaluation would additionally need the OE/board toolchain stated with the
result.

<a id="command"></a>
## Command

Host checks (cwd: repository root; success: all tests OK, exit 0):

```bash
python3 -m unittest discover -s samples/vision/efficientnet/tests -v
```

Functional board check on X5 (prerequisite:
`bash samples/vision/efficientnet/model/download.sh x5 b2`; success: exit 0
and a Top-5 containing a deerhound-related class):

```bash
python3 samples/vision/efficientnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientnet:EfficientNet_B2_224x224_nv12.bin \
  --model-path samples/vision/efficientnet/model/EfficientNet_B2_224x224_nv12.bin \
  --test-img samples/vision/efficientnet/test_data/Scottish_deerhound.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

On S100/S600 substitute the `s:` reference and the `s100/`/`s600/`
artifact path (for example lite1 at 240x240:
`s:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm`); the labels
file is shared. For a same-board before/after comparison, run the legacy
platform entrypoint
(`platforms/x5/samples/vision/efficientnet/runtime/python/main.py` or
`platforms/s/samples/vision/efficientnet/runtime/python/main.py`) with
the same image, model bytes, labels, resize type, and Top-K, and compare
class IDs and Top-K scores before label formatting (identical IDs; scores judged within a stated tolerance — the recorded 2026-09-21 smoke used abs diff < 1e-5; raw-tensor equality was not asserted). Comparing X5 against S
is not a substitute for a same-board before/after comparison. The output
should be finite, non-zero, and stable across repeated runs with the same
input.

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| contract pass | runtime accepts the artifact, tensor names/shapes/dtypes match the binding, one F32 score vector returns | any prepared artifact on its matching board |
| Top-K agreement | identical post-softmax Top-K class IDs between canonical and legacy runs; scores within tolerance (2026-09-21 smoke: max abs diff <=1.2e-7 against the 1e-5 bar) | same board, same artifact bytes, image, resize type, Top-K |
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
| host tests | 28 OK (2026-09-21; 25 at batch completion, +3 B2-R1 remediation regression tests, author self-check) | migration evidence |
| board comparison (canonical vs legacy) | passed (2026-09-21: x5-8g/x5-4g b2/b3/b4 and s100/s600 lite0..lite4 top-5 ids exactly equal, max abs diff <=1.2e-7; run.sh rc=0 on all four boards; omitted-variant default entry re-verified on s100/s600 after B2-R1; s100p explicit rejection) | [B2 board evidence](../../../../docs/releases/unified-migration/evidence/2026-09-21-b2-board-smoke-evidence.json) |
| dataset accuracy / latency | not-run in this sample | — |

Published historical figures, not re-measured in this repository.

X5 source release (rdk_x5 @ac11571, x5-v1.1.3; source notes: Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model,
latency single-frame single-thread single-core, FPS multi-threaded;
CPU 8xA55@1.8GHz performance mode, BPU 1xBayes-e@1GHz):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread Latency (ms) | Multi-thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 19.27 | 74.25% | 71.75% | 5.44 | 18.63 | 212.75 |
| EfficientNet-B3 | 224x224 | 12.19 | 76.22% | 74.05% | 3.96 | 12.76 | 310.30 |
| EfficientNet-B2 | 224x224 | 9.07 | 76.50% | 73.25% | 3.31 | 10.51 | 376.77 |

S source release (rdk_s @380e1a2, s-v1.1.2; conditions unstated beyond
the table):

| Variant | Single-thread latency | Single-thread FPS | Multi-thread latency | Multi-thread FPS |
| --- | --- | --- | --- | --- |
| Lite0 | 0.448 ms | 2107.815 | 0.591 ms | 4827.886 |
| Lite1 | 0.489 ms | 1948.957 | 0.708 ms | 4086.470 |
| Lite2 | 0.565 ms | 1702.519 | 0.935 ms | 3123.682 |
| Lite3 | 0.668 ms | 1451.031 | 1.249 ms | 2345.518 |
| Lite4 | 0.915 ms | 1064.339 | 1.979 ms | 1487.055 |

<a id="boundaries"></a>
## Boundaries

No dataset-level accuracy or latency harness ships with this sample: the
checked-in material covers host contract tests and functional board
checks only. Host test success never certifies a board. A board that is
unreachable or an artifact that is unavailable makes the corresponding
item `not-run`, not failed-and-forgotten.
