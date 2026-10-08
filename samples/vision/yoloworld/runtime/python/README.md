# YOLOWorld Python runtime

English | [简体中文](README_cn.md)

<a id="overview"></a>
## Python inference

Run YOLOWorld detection using the bundled offline vocabulary embeddings and per-image prompts. `YOLOWorldTask` initializes the runtime and returns detections with the matching prompt context.

<a id="directory"></a>
## Directory structure

```text
python/
├── cli.py  # Arguments, model selection and result presentation
├── main.py  # Command-line entry: construct the model and call predict
├── model_binding.py  # Model selection and physical tensor contracts
├── model_runner.py  # Runtime loading and raw tensor execution
├── run.sh  # Locate the Python entry and forward arguments
└── yoloworld.py  # Model stages and prediction
```

<a id="environment"></a>
## Environment

Use the X5 system Python and BSP-provided `hbm_runtime`, with NumPy, OpenCV and PyYAML installed. Follow [model preparation](../../model/README.md) before inference. Run the commands below from the repository root.

<a id="usage"></a>
## Usage

Default invocation:

```bash
python3 samples/vision/yoloworld/runtime/python/main.py --target x5 --prompts dog
```

SDK-free protocol inspection:

```bash
python3 samples/vision/yoloworld/runtime/python/main.py --dry-run --target x5 --prompts dog
```

Custom paths require the exact identity:

```bash
python3 samples/vision/yoloworld/runtime/python/main.py --target x5 --model-path /absolute/yolo_world.bin --asset-id x5:yoloworld:yolo_world.bin --vocab-file /absolute/offline_vocabulary_embeddings.json --test-img /absolute/dog.jpeg --prompts dog,person --img-save-path /absolute/result.png --priority 0 --bpu-cores 0
```

`--prompts` rejects empty items and unknown vocabulary words. `--score-thres`
defaults to `0.05`; `--nms-thres` defaults to `0.45`; `--priority` defaults to
0 and `--bpu-cores` defaults to `[0]`.

<a id="parameters"></a>
## Parameters

`--target` must be explicit for dry-run and is checked against the board before
runtime load. `--model-path` and `--asset-id` identify the exact model. The
vocabulary JSON and image paths default to the sample's `test_data` files.
Prompts are per-call context; at most 32 nonempty entries are accepted, and the
last prompt fills remaining text slots. No label-file fallback exists.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--target` | str | `auto` | Requested target; dry-run requires explicit `x5`. |
| `--model-path` | path | `None` | External model path; requires exact asset ID. |
| `--asset-id` | str | `None` | Exact `x5:yoloworld:yolo_world.bin` identity. |
| `--vocab-file` | path | samples/vision/yoloworld/test_data/offline_vocabulary_embeddings.json | Offline F32 embeddings. |
| `--test-img` | path | samples/vision/yoloworld/test_data/dog.jpeg | BGR input image. |
| `--prompts` | str | `dog` | Nonempty comma-separated vocabulary words. |
| `--img-save-path` | path | samples/vision/yoloworld/test_data/inference.png | Annotated output image. |
| `--priority` | int | `0` | Runtime priority 0..255. |
| `--bpu-cores` | int list | `[0]` | Runtime BPU core indexes. |
| `--score-thres` | float | `0.05` | Score filter. |
| `--nms-thres` | float | `0.45` | Class-wise IoU threshold. |
| `--list-models` | flag | `false` | Print manifest identity without SDK. |
| `--dry-run` | flag | `false` | Print protocol without board/SDK. |

<a id="results"></a>

<a id="integration-example"></a>
## Results and integration example

```python
import cv2, json
from samples.vision.yoloworld.runtime.python.model_binding import resolve_selection
from samples.vision.yoloworld.runtime.python.yoloworld import YOLOWorldTask
selection = resolve_selection('x5')
with open('samples/vision/yoloworld/test_data/offline_vocabulary_embeddings.json') as f:
    vocabulary = json.load(f)
task = YOLOWorldTask(selection, vocabulary)
image = cv2.imread('samples/vision/yoloworld/test_data/dog.jpeg')
prepared = task.preprocess(image, ['dog'])
raw = task.infer(prepared)
explicit_result = task.postprocess(raw, prepared.context)
composed_result = task.predict(image, ['dog'])
```

`explicit_result` and `composed_result` are `DetectionResult` values with
`boxes[N,4]`, `scores[N]`, `class_ids[N]`, and the prompt tuple. The example
requires an X5 and the prepared model.

<a id="stage-io"></a>
## Stage I/O

| Stage | Input | Output and responsibility |
| --- | --- | --- |
| `preprocess` (alias `pre_process`) | BGR HxWx3 uint8/F32 image, prompt sequence | immutable image F32[1,3,640,640], text F32[1,32,512,1], scale/original-shape/prompt-ID context |
| `infer` (alias `forward`) | prepared tensors | native raw F32[1,8400,32] and F32[1,8400,4], with no reshape/dequant/NMS |
| `postprocess` (alias `post_process`) | raw tensors plus that call's context | argmax slot, score threshold, class-wise NMS, coordinate scaling and result arrays |
| `predict` | image and prompts | exactly the explicit three stages for one call |

The runner validates observed metadata before execution and never changes native
output values (a source terminal singleton is accepted and preserved). The task owns source F32 preprocessing, slot filling, NMS and
coordinate conversion.

<a id="troubleshooting"></a>
## Troubleshooting

An empty prompt, unknown prompt, non-finite input, wrong metadata shape/dtype,
missing model, unknown board, or target mismatch is an error. A missing model is
fixed by running the explicit model command. The evaluator requires a new output
directory for each run.
