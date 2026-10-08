English | [简体中文](README_cn.md)

# ByteTrack Python runtime

<a id="overview"></a>
## Python inference

Detect people with YOLOv5 and associate detections across ordered video frames with ByteTrack.

<a id="directory"></a>
## Directory structure

```text
python/
├── tracker_backend/  # BYTETracker association, Kalman filtering and matching
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── TRACKER_SOURCE_MAP.json  # Structured data
├── cli.py  # Arguments, detector selection and preview
├── main.py  # CLI entry: construct model and call predict
├── run.sh  # Run the sample
└── tracking.py  # Model initialization and inference stages
```

Start with [main.py](main.py): it constructs `ByteTrackTask.from_model` and calls `predict`. [tracking.py](tracking.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

Use Python 3, NumPy, OpenCV, SciPy, `lap==0.5.12`, and `cython-bbox==0.1.5`; the latter two are used by the CPU matching backend. Board inference additionally needs S `hbm_runtime`. `--help`, `--list-models`, and explicit-target `--dry-run` do not load SDK, model, video, or network resources. The detector is S split-NV12 672x672; the tracker consumes XYXY boxes, scores, and class IDs.

<a id="usage"></a>
## Usage

The input video is not bundled. After explicit model/video preparation, run from the repository root:

```bash
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_unified.mp4
```

Success writes a nonempty output video, prints `Saved <N> tracked frames...`, and exits `0`. Add `--records...jsonl` to save every frame's track snapshots. `--max-frames` bounds a smoke run; `0` means all frames.

<a id="parameters"></a>
## Parameters

| option | type | default | meaning |
|---|---|---|---|
| `--target` | choice | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600`; only S three targets resolve |
| `--asset-id` | string | `null` | exact detector identity |
| `--model-path` | path | `null` | existing HBM; requires asset ID |
| `--input` | path | `samples/vision/bytetrack/test_data/track_test.mp4` | user-prepared video; not bundled in the repository |
| `--output` | path | `samples/vision/bytetrack/test_data/result_unified.mp4` | output video |
| `--records` | path | `null` | optional JSONL frame records |
| `--score-thres` | float | `0.25` | detector confidence |
| `--nms-thres` | float | `0.45` | detector IoU NMS |
| `--track-thresh` | float | `0.3` | BYTETracker high-score threshold |
| `--track-buffer` | int | `60` | lost-frame buffer |
| `--match-thresh` | float | `0.8` | first association threshold |
| `--frame-rate` | int | `30` | tracker frame-rate scaling |
| `--mot20` | flag | `false` | MOT20 matching mode (plain 1 − IoU cost, no score fusion) |
| `--priority` | int | `0` | scheduler priority |
| `--bpu-cores` | int list | `[0]` | BPU core indexes |
| `--max-frames` | int | `0` | 0 processes all frames |
| `--list-models` | flag | `false` | manifest-only |
| `--dry-run` | flag | `false` | explicit target; no SDK/model/video |

`--list-models` and `--dry-run` are mutually exclusive. User errors return `2`.

<a id="results"></a>
## Results

The CLI writes an MP4 annotated with person tracks and optional JSONL rows `{frame, tracks:[{track_id, tlbr, score, frame_id}]}`. `ByteTrackTask.postprocess` (alias `post_process`) filters class `0`, removes non-positive-width/height boxes before tracker update, and returns owned immutable `Track` snapshots. Empty detections still call `tracker.update` and advance frame state. IDs are process-global and monotonic; a fresh process starts its own counter.

<a id="integration-example"></a>
## Integration example

After the exact HBM is prepared and the CPU dependencies are installed, this complete example defines all variables. It processes two frames in order and shows explicit stages:

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.bytetrack.runtime.python.cli import resolve_selection
from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask, TrackingConfig

target = "s100"
asset_id = "s:bytetrack:s100/yolov5x_672x672_nv12.hbm"
model_path = Path("samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm")
selection = resolve_selection(target, asset_id=asset_id, model_path=model_path)
task = ByteTrackTask.from_model(
    selection, config=TrackingConfig(), score_thres=0.25, nms_thres=0.45
)
task.detector.set_scheduling_params(priority=0, bpu_cores=[0])
frame_path = Path("samples/vision/bytetrack/test_data/bus.jpg")
frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
if frame is None:
    raise FileNotFoundError(frame_path)
prepared = task.preprocess(frame)
native_outputs = task.infer(prepared.tensors)
explicit_result = task.postprocess(native_outputs, prepared.context)
composed_result = task.predict(frame)
assert isinstance(explicit_result, tuple) and isinstance(composed_result, tuple)
print(explicit_result, composed_result)
```

`predict` is stateful and should be called once per ordered frame; do not compare a second call to the first as if it were a pure function.

<a id="stage-io"></a>
## Stage I/O

- `preprocess(frame)` (alias `pre_process`) delegates the detector and returns its tensors plus immutable geometry context.
- `infer(tensors)` (alias `forward`) delegates native detector inference and does not update tracker state.
- `postprocess(outputs, context)` (alias `post_process`) decodes the detector, filters person class, removes invalid clipped boxes, updates tracker once, and returns `tuple[Track,...]`.
- `predict(frame)` composes one ordered frame. `reset` creates a fresh stream with `frame_index == 0` while preserving the process-global ID counter.

<a id="troubleshooting"></a>
## Troubleshooting

- Missing video is an explicit error; prepare `track_test.mp4` first. No fallback download occurs.
- A custom HBM path without its exact qualified asset ID is rejected.
- `--dry-run` with `auto` is rejected because execution target must be explicit.
- Letterbox padding can clip a box to zero area; the task drops such boxes, and the evaluator treats any non-finite record as a failed capture.
- Do not share one task between independent videos or threads; tracker state is intentionally not thread-safe.
