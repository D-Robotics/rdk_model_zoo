# RepViT Python runtime

`main.py` is the user-facing command: it parses arguments,
constructs the model, calls `predict`, and shows the result. The complete
classification flow lives in [`classify.py`](classify.py):
`RepViTClassifier` shows initialization, `preprocess`, `infer`, `postprocess` and
`predict` in one readable file, reusing the shared NV12 packing, Top-K
math and lazy runner.

<a id="overview"></a>
## Python inference

Use this directory for python inference.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── __init__.py  # Python script
├── classify.py  # Classification preprocessing, inference, and postprocessing
├── cli.py  # Arguments and result presentation
├── main.py  # Command-line entry
├── model_binding.py  # Python script
├── model_runner.py  # Python script
└── run.sh  # Run the sample
```

<a id="environment"></a>
## Environment

Use X5 with its matching `hbm_runtime`, NumPy, OpenCV and PyYAML. Host imports/help/list/dry-run need no board SDK. Host test dependencies: `samples/vision/repvit/requirements-host.txt`.

[Full prerequisites and tested host versions](../../README.md#prerequisites). For board inference, use the matching board image with `hbm_runtime`.

<a id="usage"></a>
## Usage

All commands use repository-root cwd. On a host:

```bash
python3 samples/vision/repvit/runtime/python/main.py --list-models
python3 samples/vision/repvit/runtime/python/main.py --dry-run --target x5
```

Expected: 3 references or default `m0_9` contract, exit 0. A prepared X5 runs:

```bash
# cwd: repository root
bash samples/vision/repvit/model/download.sh x5 m0_9
python3 samples/vision/repvit/runtime/python/main.py \
  --target x5 --variant m0_9 \
  --test-img samples/vision/repvit/test_data/yurt.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`run.sh` forwards all runtime options. Download is a separate action.

After preparing the model above, run the default entry on X5 (target detected automatically):

```bash
# cwd: repository root
python3 samples/vision/repvit/runtime/python/main.py
```

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | choice | auto | execution target: `auto`, `x5`, `s100`, `s100p`, `s600`; an execution target must match detected hardware |
| `--asset-id` | string | null | complete `group:sample:filename` reference from the manifest |
| `--variant` | choice | null | model variant (`m0_9` default when omitted; see `--list-models` for published combinations) |
| `--model-path` | string | null | existing `.bin`; must be paired with `--asset-id`; defaults to the `model/` location for the resolved reference when omitted |
| `--test-img` | string | samples/vision/repvit/test_data/yurt.JPEG | BGR input image |
| `--label-file` | string | datasets/imagenet/imagenet_classes.names | one-label-per-line ImageNet labels |
| `--top-k` | int | 5 | number of printed results |
| `--topk` | int | 5 | alias of `--top-k` |
| `--resize-type` | int | null | `0` direct stretch or `1` letterbox with BGR 127 padding; default follows the bound source (1) |
| `--priority` | int | 0 | runtime scheduling priority (0-255) |
| `--bpu-cores` | int list | [0] | runtime BPU core indexes |
| `--img-save-path` | string | null | optional annotated output image path |
| `--list-models` | flag | false | list manifest-backed references without board access |
| `--dry-run` | flag | false | resolve/check a selection without loading a model or SDK |

<a id="results"></a>
## Results

Prints Top-K class ID/score/label; `ClassificationResult` carries int64 IDs, float32 scores and a label tuple. Pass `--img-save-path` to save a visualization; otherwise the runtime prints results to stdout. Score semantics are raw logits plus softmax; equal scores use ascending
ID order.

<a id="integration-example"></a>
## Integration example

cwd: repository root. Prepare the artifact with the downloader before constructing the classifier. Labels are optional and omitted here, so their values are class ID strings.

```python
from samples.vision.repvit.runtime.python.classify import RepViTClassifier
from samples.vision.repvit.runtime.python.model_binding import resolve_selection

selection = resolve_selection("x5", variant="m0_9")
model = RepViTClassifier(selection, top_k=5)
result = model.predict("samples/vision/repvit/test_data/yurt.JPEG")
print(result.class_ids, result.scores, result.labels)
```

`predict` accepts a local image path or BGR `uint8` array; the input array remains unchanged. The shared
`ClassificationTask` flow stays importable from
[`classification.py`](../../../../../utils/py_utils/classification.py).

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output |
| --- | --- | --- |
| preprocess (pre_process) | image path or BGR uint8 H×W×3 | PreparedInput: named flat NV12 uint8 tensor (75264 bytes) and immutable resize context |
| infer (forward) | PreparedInput | runner raw output mapping, unchanged; no softmax or sorting |
| postprocess (post_process) | F32 scores squeezing to (1000,) | softmax and stable Top-K ClassificationResult |
| predict | image path or BGR image | the three stages chained |

Default preprocessing is linear letterbox with BGR 127 padding. Classification postprocessing does not consume geometry. The runner handles SDK loading, scheduling and metadata checks; file/label loading and drawing stay in the CLI layer (`cli.py`).

<a id="troubleshooting"></a>
## Troubleshooting

Prepare the target-specific artifact with the model downloader. When using `--model-path`, pass its exact qualified reference with `--asset-id`. Select a target listed in the support matrix and check that artifact metadata matches the sample tensor contract before inference.
