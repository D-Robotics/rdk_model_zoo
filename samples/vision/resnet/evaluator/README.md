# ResNet18 evaluation

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
data (50,000 images, ILSVRC2012 val) prepared separately by the user; no
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
python3 -m unittest discover -s samples/vision/resnet/tests -v
```

Functional board check on X5 (prerequisite:
`bash samples/vision/resnet/model/download.sh x5`; success: exit 0 and the
expected Top-K):

```bash
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file platforms/x5/datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

On S100/S600 substitute the `s:resnet18:<target>/...` reference, artifact
path, and `platforms/s/...` labels. For a same-board before/after
comparison, run the legacy entrypoint
(`platforms/x5/samples/vision/resnet/runtime/python/main.py` or
`platforms/s/samples/vision/resnet18/runtime/python/main.py`) with the
same image, model bytes, labels, resize type, and Top-K, and compare class
IDs and raw scores before label formatting. The S-series C++ check runs
`bash samples/vision/resnet/runtime/cpp/run.sh`.

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
| host tests | 39 OK (2026-09-20, host evidence JSON) | migration evidence |
| board comparison | canonical == legacy on both X5 boards and S100 (class IDs and raw scores) | integration review 2026-09-17 |
| S100 C++ | Top-5 text equal to the source baseline | integration review 2026-09-17 |
| dataset accuracy / latency | not-run in this sample | — |

Historical legacy figures (X5 evaluator record): Top-1 71.5% (float) /
70.5% (quantized), 2.95 ms latency, 449+ FPS. The source does not state
whether latency/FPS used single calls, batching, or threads; they are not
re-derived here and not presented as results of the canonical sample.

<a id="boundaries"></a>
## Boundaries

No dataset-level accuracy or latency harness ships with this sample: the
checked-in material covers host contract tests and functional board
checks only. Host test success never certifies a board. A board that is
unreachable or an artifact that is unavailable makes the corresponding
item `not-run`, not failed-and-forgotten. S600 re-validation remains
`not-run` until board access recovers.
