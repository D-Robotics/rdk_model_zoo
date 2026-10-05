# ViT Python runtime

`main.py` is the canonical user-facing command: it parses arguments,
constructs the model, calls `predict`, and shows the result. The complete
classification flow lives in [`classify.py`](classify.py):
`ViTClassifier` shows initialization, `preprocess`, `infer`, `postprocess` and
`predict` in one readable file, reusing the shared NV12 packing, Top-K
math and lazy runner.

<a id="environment"></a>
## Environment

Full checkout required. Host tested: Python 3.14.7, NumPy 2.5.3, OpenCV 4.14.0, PyYAML 6.0.3. S100 inference requires its board-provided `hbm_runtime`; exact board image/SDK/Python versions, minimum RAM and disk capacity remain unverified. Allow disk space for the checkout, selected HBM and outputs. OE is only needed for conversion.

```bash
# cwd: repository root
python3 -m venv .venv-vit
source .venv-vit/bin/activate
python3 -m pip install -r samples/vision/vit/requirements-host.txt
```

<a id="usage"></a>
## Usage

Prepare the default artifact using the root Quick Start. On S100, the zero-argument entry detects board identity. All commands below use repository-root cwd; success is exit 0, Top-5 output.

```bash
python3 samples/vision/vit/runtime/python/main.py
python3 samples/vision/vit/runtime/python/main.py --target s100 --variant int8 --test-img samples/vision/vit/test_data/airplane_0000.png --label-file samples/vision/vit/test_data/cifar10_classes.names --top-k 5
```

Host-only inspection:

```bash
python3 samples/vision/vit/runtime/python/main.py --list-models
python3 samples/vision/vit/runtime/python/main.py --dry-run --target s100 --variant int16
```

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--target` | choice | auto | auto/x5/s100/s100p/s600; only s100 has assets |
| `--asset-id` | str | None | exact manifest reference |
| `--variant / --model-variant` | choice | None | int8/int16; resolved default int8 |
| `--model-path` | str | None | override requires asset-id |
| `--test-img` | str | samples/vision/vit/test_data/airplane_0000.png | BGR image |
| `--label-file` | str | samples/vision/vit/test_data/cifar10_classes.names | dictionary or one label per line |
| `--top-k / --topk` | int | 5 | 1..10 |
| `--resize-type` | choice | None | 0 direct nearest (binding default); 1 linear letterbox |
| `--priority` | int | 0 | 0..255 |
| `--bpu-cores` | int list | [0] | board core indexes |
| `--img-save-path` | str | None | optional annotated image |
| `--list-models` | flag | false | no SDK or download |
| `--dry-run` | flag | false | selection only; no inference |

Image/label defaults resolve to absolute paths inside the checkout. `None` for variant/resize means the bound source default, not a missing setting. List/dry-run are mutually exclusive.

<a id="results"></a>
## Results

ClassificationResult contains class_ids (integer array), scores (softmax probabilities) and labels (tuple). IDs 0–9 follow the bundled CIFAR mapping. CLI prints them; optional image path receives an annotated copy. No default output file.

<a id="integration-example"></a>
## Integration example

Board example, after preparing int8; imports do not download anything.
The same pipeline is exercised with an injected host runner in tests.

```python
# cwd: repository root on S100; prepare int8 first
from pathlib import Path
from samples.vision.vit.runtime.python.classify import ViTClassifier
from samples.vision.vit.runtime.python.model_binding import resolve_selection
from samples.vision.vit.runtime.python.labels import load_labels

selection = resolve_selection("s100", variant="int8")
labels = load_labels(Path("samples/vision/vit/test_data/cifar10_classes.names"))
model = ViTClassifier(selection, top_k=5, labels=labels)
model.set_scheduling_params(priority=0, bpu_cores=[0])
result = model.predict("samples/vision/vit/test_data/airplane_0000.png")
print(result.class_ids, result.scores, result.labels)
```

`predict` accepts a local image path or a BGR `uint8` array. The stages
can also be driven explicitly (`preprocess`/`infer`/`postprocess`); the
established `pre_process`/`forward`/`post_process` spellings are thin
aliases, and the shared `ClassificationTask` stays importable from
[`classification.py`](classification.py).

<a id="stage-io"></a>
## Stage I/O

preprocess (pre_process): image path or BGR U8 H×W×3 → PreparedInput with Y U8 [1,224,224,1], UV U8 [1,112,112,2] and per-call geometry. Direct resize uses nearest; letterbox uses linear and padding 127. infer only calls the runner, preserving the raw mapping. postprocess squeezes a F32 ten-score vector, applies stable softmax and selects Top-K; no file or SDK access. predict composes these stages; the established pre_process/forward/post_process spellings are thin aliases. Runtime metadata validates shapes/dtypes before run; quantized raw outputs are rejected rather than silently reinterpreted.

<a id="troubleshooting"></a>
## Troubleshooting

Missing HBM: run explicit model preparation. Target mismatch: execute on S100, do not override an S100P identity. Wrong class count/shape/dtype: verify artifact reference and metadata; do not rename a model to bypass validation. Empty/unreadable image or invalid Top-K: fix the input/1..10 argument. External model path requires exact asset-id.
