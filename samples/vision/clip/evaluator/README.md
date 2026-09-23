English | [简体中文](./README_cn.md)

# Evaluator — CLIP image-text matching

This directory records the source validation path. The source provides no published benchmark table, so this document does not invent latency or accuracy values. Board and model execution were not performed in this migration.

<a id="dataset"></a>
## Dataset

The default validation input is `samples/vision/clip/test_data/dog.jpg` with prompts `a diagram` and `a dog`. The source BPE vocabulary is the real `platforms/x5/samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz` (preserved in the unified runtime). No larger evaluation dataset or preparation script is published.

```text
# cwd: repository root
samples/vision/clip/test_data/dog.jpg
samples/vision/clip/runtime/python/bpe_simple_vocab_16e6.txt.gz
# prompts: a diagram,a dog
```

<a id="environment"></a>
## Environment

- Target: RDK X5; image encoder through board `hbm_runtime`, text encoder through CPU `onnxruntime`.
- Host tests: Python 3.14.7, NumPy, OpenCV, `ftfy==6.3.1`, and `regex==2026.9.10`; injected runtime fixtures avoid an ONNX Runtime requirement on the host.
- Legacy comparison source directory: `platforms/x5/samples/vision/clip/runtime/python`. It uses the source `SimpleTokenizer` and BPE file; the legacy main entrypoint is not invoked because it overwrites `test_data/inference.png`.

<a id="command"></a>
## Evaluation Command

The source validation entrypoints are `runtime/python/main.py` and `run.sh`. The explicit command below runs the current sample and writes a user-selected image. It is documented only and was not run here.

```bash
# cwd: repository root; prerequisites: both model assets prepared for X5
python3 samples/vision/clip/runtime/python/main.py \
  --target x5 \
  --test-img samples/vision/clip/test_data/dog.jpg \
  --texts "a diagram,a dog" \
  --img-save-path /tmp/clip-eval/inference.png
# expect: JSON scores/order and /tmp/clip-eval/inference.png; no published numeric benchmark
```

For a same-board legacy/unified raw comparison, use the real legacy directory and a unique UTC output folder. This procedure reuses the source BPE through `CLIPMatcher`; it does not invoke the legacy main entrypoint or overwrite `test_data/inference.png`. It preserves complete raw image/text arrays and asserts parity. This procedure was not run here.

```bash
# cwd: repository root; prerequisites: X5 board, exact model pair, and board on which both runtimes are available
PYTHONPATH="$PWD:$PWD/platforms/x5/samples/vision/clip/runtime/python" python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

from samples._shared.platforms import require_execution_target
require_execution_target("x5")  # before legacy imports or either SDK factory
from clip_retrieval import CLIPConfig, CLIPMatcher
from samples.vision.clip.runtime.python.matching import CLIPTask
from samples.vision.clip.runtime.python.model_binding import resolve_selection
from samples.vision.clip.runtime.python.model_runner import RuntimeModelRunner
from samples.vision.clip.runtime.python.tokenization import PromptTokenizer

repo = Path.cwd()
image_path = repo / "samples/vision/clip/test_data/dog.jpg"
image_model_path = repo / "samples/vision/clip/model/img_encoder.bin"
text_model_path = repo / "samples/vision/clip/model/text_encoder.onnx"
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
texts = ["a diagram", "a dog"]
if image is None or not image_model_path.is_file() or not text_model_path.is_file():
    raise RuntimeError("prepare the image and exact X5 model pair before running")

run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
raw_dir = repo / "evaluator-output" / f"clip-raw-{run_id}"
raw_dir.mkdir(parents=True, exist_ok=False)

legacy = CLIPMatcher(CLIPConfig(str(image_model_path), str(text_model_path)))
legacy_inputs = legacy.pre_process(image)
legacy_image_nested = legacy.image_model.run(legacy_inputs)
legacy_image_raw = np.asarray(legacy_image_nested[legacy.image_model_name][legacy.output_names[0]])
legacy_text_raw = np.asarray(legacy.text_session.run(
    [legacy.text_output_name], {legacy.text_input_name: legacy.tokenize(texts)}
)[0])
np.save(raw_dir / "legacy_image.npy", legacy_image_raw, allow_pickle=False)
np.save(raw_dir / "legacy_text.npy", legacy_text_raw, allow_pickle=False)

selection = resolve_selection("x5")
runner = RuntimeModelRunner(selection)
binding = runner.load()
task = CLIPTask(runner, binding, PromptTokenizer())
prepared = task.pre_process(image, texts)
unified_raw = task.forward(prepared.tensors)
np.save(raw_dir / "unified_image.npy", unified_raw["image_feature"], allow_pickle=False)
np.save(raw_dir / "unified_text.npy", unified_raw["text_features"], allow_pickle=False)

np.testing.assert_array_equal(legacy_inputs[legacy.image_model_name][legacy.input_names[0]],
                              prepared.tensors["image"])
np.testing.assert_array_equal(legacy.tokenize(texts), prepared.tensors["texts"])
for reference, candidate in ((legacy_image_raw, unified_raw["image_feature"]),
                             (legacy_text_raw, unified_raw["text_features"])):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError((reference.shape, reference.dtype, candidate.shape, candidate.dtype))
    np.testing.assert_allclose(reference, candidate, rtol=0.0, atol=1e-5)
print({"raw_dir": str(raw_dir), "comparison": "passed", "run_id": run_id})
PY
# expect: four complete raw .npy arrays in a unique UTC directory; assertion failure exits nonzero
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target` | str | `x5` | The only published CLIP target. |
| `texts` | list[str] | `a diagram`, `a dog` | Source prompts encoded with the preserved BPE vocabulary. |
| `run_id` | UTC string | generated per run | Prevents raw-array overwrites. |
| `raw_dir` | path | `evaluator-output/clip-raw-<run_id>` | Unique directory containing complete legacy/unified image and text arrays. |

<a id="metrics"></a>
## Metrics

| Metric | Definition | Conditions |
| --- | --- | --- |
| Cosine similarity | Dot product of each text feature with the image feature divided by both L2 norms plus `1e-12`. | One image, N prompts, float32 features; scores remain in prompt order. |
| Rank order | Descending `argsort` of cosine scores. | Retains source NumPy argsort ordering; no cross-version tie-order guarantee. |
| Legacy parity | Raw image/text encoder arrays compared before cosine conversion. | Same board, same input, same source BPE; identical input tensors, exact shape and dtype F32, `rtol=0`, `atol=1e-5`. |

No latency, top-k, retrieval, or classification metric is published for this sample.

<a id="outputs"></a>
## Outputs

The runtime writes the annotated image exactly at `--img-save-path` and prints JSON scores/order. The optional comparison writes `legacy_image.npy`, `legacy_text.npy`, `unified_image.npy`, and `unified_text.npy` under a unique UTC directory. No raw arrays or benchmark result from this migration exist.

<a id="reference-results"></a>
## Reference Results

There is no published numeric benchmark table in the source evaluator. The source validation expectation is qualitative: for `dog.jpg`, the score for `a dog` should exceed the score for `a diagram`. This migration status is `not-run`; no numeric value is asserted.

| Reference | Value | Conditions | Source |
| --- | --- | --- | --- |
| Dog prompt ranking | `a dog` ranks above `a diagram` | X5 pair, source BPE, `dog.jpg`, cosine ranking | `platforms/x5/samples/vision/clip/evaluator/README.md` |
| Published benchmark | not provided | no source latency/accuracy table | source evaluator README |

<a id="boundaries"></a>
## Boundaries

- No standalone dataset evaluator or published benchmark is provided.
- The text encoder is CPU ONNX and requires ONNX Runtime on the board; host fixture tests do not prove board availability.
- The legacy comparison is raw-array parity only and does not run the legacy main entrypoint or overwrite the tracked visualization.
- This sample covers image-text similarity only; no C++ path, training, or text-generation task is included.

## License

Evaluator documentation follows the repository [LICENSE](../../../../LICENSE), Apache-2.0.
