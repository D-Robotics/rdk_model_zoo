English | [简体中文](README_cn.md)

# Python Runtime — CLIP image-text matching

<a id="overview"></a>
## Python inference

Encode an image and text prompts with CLIP, then compare their feature similarity.

<a id="directory"></a>
## Directory structure

```text
python/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── cli.py  # Arguments, model selection and result output
├── main.py  # CLI entry: construct model and call predict
├── matching.py  # Model initialization and inference stages
├── run.sh  # Run the sample
├── simple_tokenizer.py  # CLIP BPE tokenization algorithm
└── tokenization.py  # Text tokenization and input preparation
```

Start with [main.py](main.py): it constructs `CLIPMatcher` and calls `predict`. [matching.py](matching.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

- Execution target: RDK X5 with `hbm_runtime` for the image encoder and `onnxruntime` with `CPUExecutionProvider` for the text encoder; choose the board image and firmware for your deployment.
- Python dependencies: Python 3.10+, NumPy, OpenCV, PyYAML, `ftfy==6.3.1`, and `regex==2026.9.10`.
- Runtime assets: `bpe_simple_vocab_16e6.txt.gz` is loaded by `PromptTokenizer`; the image and text model files are prepared separately. BPE cleaning rules and token IDs follow the original vocabulary.

<a id="usage"></a>
## Usage

Prepare both assets using [`../../model/README.md`](../../model/README.md), then run on X5 from the repository root:

```bash
# cwd: repository root; model: img_encoder.bin and text_encoder.onnx already prepared
python3 samples/vision/clip/runtime/python/main.py --target x5
# success: exit code 0, JSON scores/order and an annotated inference.png
```

Choose a different image, prompts, and visualization destination:

```bash
# cwd: repository root; model pair: samples/vision/clip/model/; BPE: bundled vocabulary
python3 samples/vision/clip/runtime/python/main.py \
  --target x5 \
  --test-img samples/vision/clip/test_data/dog.jpg \
  --texts "a diagram,a dog" \
  --img-save-path /tmp/clip-inference.png
# success: exit code 0, JSON scores/order, and /tmp/clip-inference.png
```

The default `--img-save-path` is the tracked `test_data/inference.png`, so a default run overwrites that file. `run.sh` only forwards arguments to `main.py`; it does not download models.

All default asset, image, and visualization paths are absolute paths derived from the sample location. Paths shown below identify their repository locations. A user-supplied relative path is resolved from the current working directory.

<a id="parameters"></a>
## Parameters

`--list-models` and `--dry-run` are mutually exclusive. Missing assets, invalid prompts, unsupported targets, and runtime errors exit 2.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | `auto`, `x5`, `s100`, `s100p`, or `s600`; only X5 has a published pair. |
| `--image-asset-id` | str | `None` | Exact image asset identity; required with `--image-model-path`. |
| `--text-asset-id` | str | `None` | Exact text asset identity; required with `--text-model-path`. |
| `--image-model-path` | str | `None` | Explicit local image `.bin` path; defaults to the manifest `img_encoder.bin`. |
| `--text-model-path` | str | `None` | Explicit local text `.onnx` path; defaults to the manifest `text_encoder.onnx`. |
| `--test-img` | str | `samples/vision/clip/test_data/dog.jpg` | Input BGR image path; parser resolves the bundled default from the sample file. |
| `--texts` | str | `a diagram,a dog` | Comma-separated prompts; empty items are removed. |
| `--img-save-path` | str | `samples/vision/clip/test_data/inference.png` | Exact annotated-image destination; default overwrites the source fixture. |
| `--priority` | int | `0` | BPU image encoder priority, constrained to 0–255. |
| `--bpu-cores` | int list | `[0]` | Nonnegative BPU image-encoder core indexes. |
| `--list-models` | flag | `false` | List the exact pair without loading either runtime. |
| `--dry-run` | flag | `false` | Resolve the pair with explicit `--target x5`, without board or SDK loading. |

<a id="results"></a>
## Results

The CLI JSON contains `target`, `prompts`, `scores`, `order`, and `image_saved`. `scores` is a float array in prompt order; `order` is the descending index array. `MatchResult` exposes the same `scores` and `order` fields. The image output is a BGR PNG/JPEG written exactly to `--img-save-path`. Cosine uses float32 features, L2 norms with `1e-12` stabilizers, and no softmax.

<a id="integration-example"></a>
## Integration Example

Prerequisite: prepare the X5 pair and run on an X5 board. The BPE vocabulary is loaded from the local bundled path. This example defines all paths, IDs, inputs, `CLIPMatcher(selection, tokenizer=PromptTokenizer())`, `explicit_result`, and `composed_result`, then compares every `MatchResult` field.

```python
from pathlib import Path
import cv2
import numpy as np

from samples.vision.clip.runtime.python.matching import CLIPMatcher
from samples.vision.clip.runtime.python.cli import resolve_selection
from samples.vision.clip.runtime.python.tokenization import PromptTokenizer

repo = Path.cwd()
image_path = repo / "samples/vision/clip/test_data/dog.jpg"
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError(f"cannot read {image_path}")

target = "x5"
image_asset_id = None
text_asset_id = None
image_model_path = None
text_model_path = None
texts = ["a diagram", "a dog"]
priority = 0
bpu_cores = [0]
selection = resolve_selection(
    target,
    image_asset_id=image_asset_id,
    text_asset_id=text_asset_id,
    image_model_path=image_model_path,
    text_model_path=text_model_path,
)
task = CLIPMatcher(selection, tokenizer=PromptTokenizer())
task.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

prepared = task.preprocess(image, texts)
raw_outputs = task.infer(prepared.tensors)
explicit_result = task.postprocess(raw_outputs)
composed_result = task.predict(image, texts)
np.testing.assert_array_equal(explicit_result.scores, composed_result.scores)
np.testing.assert_array_equal(explicit_result.order, composed_result.order)
print({"scores": composed_result.scores.tolist(),
       "order": composed_result.order.tolist()})
```

<a id="stage-io"></a>
## Three-Stage I/O

- `preprocess`: BGR `uint8` `H×W×3` plus nonempty text sequence → `PreparedInput`. The image is RGB bicubic resized with the short side fixed to 224 and the proportional long side rounded, center-cropped to 224, divided by 255, and emitted as contiguous float32 `image` `(1,3,224,224)`. No CLIP mean/std normalization is applied. The real BPE vocabulary produces contiguous int32 `texts` `(N,77)` with source SOT/EOT IDs. `context` stores geometry and texts.
- `infer`: semantic `image`/`texts` tensors → raw mapping `image_feature` float32 `(1,512)` and `text_features` float32 `(N,512)`. The runner adapts semantic keys to dynamic image metadata names and ONNX text names; text metadata must be I32 `[N,77]` input and F32 `[N,512]` output.
- `postprocess`: raw features → `MatchResult(scores, order)`. It computes cosine similarity and descending `argsort`; no softmax or feature L2 mutation is returned.
- `predict(image, texts)` composes exactly the three stages. The model initializes a tokenizer, or accepts one through its `tokenizer` argument. `preprocess` uses it to encode prompts; the CLI handles rendering and file output.

For application development, [cli.py](cli.py) holds option declarations, the model-free `--list-models`/`--dry-run` modes, prompt parsing and presentation, while `main.py` parses, resolves, constructs `CLIPMatcher` and calls `predict`.

<a id="troubleshooting"></a>
## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Model not found:...; prepare the pair with model/download.sh.` | One encoder asset is missing. | Run the explicit pair download or provide both exact asset IDs with paths. |
| `No published CLIP encoder pair for s100` | Only X5 has a manifest pair. | Use an X5 target. |
| `External img_encoder.bin path requires its exact asset-id.` | Custom image path has no identity. | Add `--image-asset-id x5:clip:img_encoder.bin`. |
| `External text_encoder.onnx path requires its exact asset-id.` | Custom text path has no identity. | Add `--text-asset-id x5:clip:text_encoder.onnx`. |
| `Input text is too long for context length 77` | Source BPE tokens exceed the fixed context and truncation is disabled. | Shorten the prompt; the CLI uses the source non-truncating behavior. |
| `CLIP text... metadata` | ONNX input/output names, dtype, or widths are incompatible. | Use the published text asset and inspect dynamic metadata; input is I32 `[N,77]`, output F32 `[N,512]`. |

## License

Runtime code follows the repository [LICENSE](../../../../../LICENSE), Apache-2.0. The BPE vocabulary and model assets retain their published provenance.
