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
├── TRACKER_SOURCE_MAP.json  # Tracker source attribution
├── cli.py  # Arguments, detector selection and preview
├── main.py  # CLI entry: construct model and call predict
├── run.sh  # Run the sample
└── tracking.py  # Model initialization and inference stages
```

Start with [main.py](main.py): it constructs `ByteTrackTask.from_model` and calls `predict`. [tracking.py](tracking.py) contains model initialization and inference stages; [cli.py](cli.py) handles arguments, model selection and result output. Model initialization loads the runtime, so applications can reuse one instance for repeated predictions.

<a id="environment"></a>
## Environment

Install Python 3, NumPy, OpenCV, SciPy, `lap==0.5.12`, and `cython-bbox==0.1.5`. All four boards use the system-provided `hbm_runtime`. X5 models take 640×640 packed-NV12 input; S models take 672×672 split-NV12 input.

On a computer without a board, use `--help` for options, `--list-models` for available models, or `--dry-run` with a specific `--target` to preview the configuration.

<a id="usage"></a>
## Usage

Prepare the model and video using the [quickstart](../../README.md#quickstart), then run from the repository root:

```bash
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target x5 --variant n-v7.0 \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_x5.mp4
```

S100:

```bash
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_unified.mp4
```

The program saves a video with person boxes and IDs and prints `Saved <N> tracked frames...`. Add `--records tracks.jsonl` to save each frame’s tracks. Use `--max-frames` to limit processing; `0` processes the whole video.

<a id="parameters"></a>
## Parameters

| option | type | default | meaning |
|---|---|---|---|
| `--target` | choice | `auto` | `auto`, `x5`, `s100`, `s100p`, `s600` |
| `--variant` | string | `null` | Detector variant; X5 defaults to `n-v7.0`, S targets to `x-672` |
| `--asset-id` | string | `null` | Full model identifier from the catalog |
| `--model-path` | path | `null` | existing BIN/HBM; requires asset ID |
| `--input` | path | `samples/vision/bytetrack/test_data/track_test.mp4` | user-prepared video; not bundled in the repository |
| `--output` | path | `samples/vision/bytetrack/test_data/result_unified.mp4` | output video |
| `--records` | path | `null` | optional JSONL frame records |
| `--score-thres` | float | `0.25` | detector confidence |
| `--nms-thres` | float | `0.45` | 检测框 NMS IoU 阈值 |
| `--track-thresh` | float | `0.3` | BYTETracker high-score threshold |
| `--track-buffer` | int | `60` | lost-frame buffer |
| `--match-thresh` | float | `0.8` | first association threshold |
| `--frame-rate` | int | `30` | tracker frame-rate scaling |
| `--mot20` | flag | `false` | MOT20 matching mode (plain 1 − IoU cost, no score fusion) |
| `--priority` | int | `0` | scheduler priority |
| `--bpu-cores` | int list | `[0]` | BPU core indexes |
| `--max-frames` | int | `0` | 0 processes all frames |
| `--list-models` | flag | `false` | List available models |
| `--dry-run` | flag | `false` | Preview configuration for a selected target |

`--list-models` and `--dry-run` are mutually exclusive. User errors return `2`.

<a id="results"></a>
## Results

The program writes an MP4 with person boxes and IDs. With `--records`, it also saves JSONL records in the form `{frame, tracks:[{track_id, tlbr, score, frame_id}]}`. `tlbr` contains the top-left and bottom-right coordinates in the original image.

`ByteTrackTask` tracks only COCO persons (class `0`). Returned `Track` records remain unchanged when later frames are processed. Frames without detections still update tracks and the frame count. New track IDs keep increasing within the same Python process.

<a id="integration-example"></a>
## Integration example

After preparing the model and dependencies, call `predict` to get tracks for a frame:

```python
from pathlib import Path
import cv2
from samples.vision.bytetrack.runtime.python.cli import resolve_selection
from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask, TrackingConfig

selection = resolve_selection("x5", variant="n-v7.0")
task = ByteTrackTask.from_model(
    selection, config=TrackingConfig(), score_thres=0.25, nms_thres=0.45
)
frame_path = Path("samples/vision/bytetrack/test_data/bus.jpg")
frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
if frame is None:
    raise FileNotFoundError(frame_path)
tracks = task.predict(frame)
for track in tracks:
    print(track.track_id, track.tlbr, track.score)
```

For each video frame, choose one calling pattern: call `predict` once, or call the three stages in order.

<a id="stage-io"></a>
## Stage I/O

- `preprocess(frame)` prepares detector inputs and returns tensors with scaling information for this frame.
- `infer(tensors)` runs the detection model.
- `postprocess(outputs, context)` extracts valid person boxes, updates tracks, and returns `tuple[Track, ...]`.
- `predict(frame)` runs these three steps in order.
- `reset()` clears existing tracks and restarts the frame count. New track IDs continue to increase.

<a id="troubleshooting"></a>
## Troubleshooting

- Video not found: check `--input` or download the example video from the [quickstart](../../README.md#quickstart).
- Custom model path: also pass the matching `--asset-id`.
- Configuration preview: use `--dry-run` with a specific `--target`, such as `x5` or `s100`.
- Multiple videos: create one instance per stream and process each stream’s frames in order.
