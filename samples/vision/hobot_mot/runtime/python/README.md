English | [简体中文](README_cn.md)

<a id="overview"></a>
## Python runtime

Use `HobotMotTracker` to assign track IDs to detections, or replay a JSONL file from the command line.
Python calls `libhobot_mot_capi.so` through `native.py`; build the library using the C++ runtime commands.

<a id="directory"></a>
## Directory structure

```text
runtime/python/
├── main.py    # arguments → tracker → predict → output
├── cli.py     # CLI: JSONL loading/validation, rendering, result JSON
├── track.py   # HobotMotTracker model class + config/input validation
├── native.py  # ctypes binding for libhobot_mot_capi.so
└── run.sh     # run main.py with forwarded arguments
```

<a id="environment"></a>
## Environment

- RDK X5, S100, S100P or S600 (ARM64 Linux)
- Python 3.8+, standard library only
- C++17 compiler and CMake 3.16+ to build the native library

Library lookup order: `--native-lib`, the `HOBOT_MOT_NATIVE_LIB` environment variable, then `runtime/cpp/build/libhobot_mot_capi.so`.

<a id="usage"></a>
## Usage

From the sample root, default run (included example sequence and config):

```bash
python3 model/download.py
cmake -B runtime/cpp/build -S runtime/cpp
cmake --build runtime/cpp/build
./runtime/python/run.sh
```

Custom run writing a result file:

```bash
./runtime/python/run.sh --input test_data/detections.jsonl \
    --config model/configs/iou2_kalman.json --output results.json
```

Both exit 0 on success and 2 with an `Error: ...` message on invalid input,
invalid config or a missing native library.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--input` | path | `samples/vision/hobot_mot/test_data/detections.jsonl` | detection JSONL (see Results for the record format) |
| `--config` | path | `samples/vision/hobot_mot/model/configs/iou2.json` | tracker config JSON (`tracker_type` must be `"IOU_2.0"`) |
| `--output` | path | null | also write full results as JSON |
| `--frame-width` | int | 960 | frame width used when a record carries no `image_width` |
| `--frame-height` | int | 544 | frame height used when a record carries no `image_height` |
| `--native-lib` | path | null | explicit path of `libhobot_mot_capi.so` |

<a id="results"></a>
## Results

Per frame one line is printed:

```text
frame <frame_index> ts <timestamp_ms>: [cls <class_id> id <track_id> (<x1>,<y1>,<x2>,<y2>) score <score> <state>] ... [removed cls <class_id> id <track_id>] ...
```

Each input JSONL line represents one frame. For example:

```json
{"frame_index": 0, "timestamp_ms": 1000, "image_width": 640, "image_height": 480, "detections": [{"bbox": [100, 100, 160, 260], "score": 0.95, "class_id": 0}]}
```

`frame_index` and the millisecond `timestamp_ms` must increase. Coordinates are pixels. Image dimensions are optional and default to the CLI frame size. Use `"detections": []` for a frame without detections.

`--output` saves a JSON array with one element per frame:

```json
[{"frame_index": 0, "timestamp_ms": 1000, "tracks": [{"class_id": 0, "track_id": 1, "bbox": [100, 100, 160, 260], "score": 0.95, "state": "valid"}], "disappeared": []}]
```

`tracks` preserves input detection order. `valid` means a track ID was assigned; `invalid` means unmatched in this frame, with ID -1. IDs are assigned separately for each class; identify a target by `(class_id, track_id)`. `disappeared` lists tracks removed in this frame after the timeout.

<a id="integration-example"></a>
## Integration example

```python
import sys
sys.path.insert(0, "samples/vision/hobot_mot/runtime/python")
from track import Detection, HobotMotTracker

tracker = HobotMotTracker(
    config_path="samples/vision/hobot_mot/model/configs/iou2.json",
    frame_width=640, frame_height=480)
with tracker:  # releases the native instance on exit
    ts_ms = 1000
    for boxes in [
        [Detection((100, 100, 160, 260), 0.95, 0)],       # person appears
        [Detection((104, 100, 164, 260), 0.95, 0)],       # person moves
        [],                                              # empty frame ages
        [Detection((112, 100, 172, 260), 0.95, 0)],       # same identity
    ]:
        result = tracker.predict(boxes, ts_ms)
        print([(t.track_id, t.state) for t in result.tracks])
        ts_ms += 40
# -> [(1, 'valid')], [(1, 'valid')], [], [(1, 'valid')]
```

<a id="stage-io"></a>
## Staged calls

- `preprocess(detections, timestamp_ms, frame_width=None, frame_height=None) -> PreparedFrame`: prepare detections, timestamp and image size for this frame.
- `infer(prepared) -> RawMotFrame`: run tracking and return IDs, boxes, states and IDs removed in this frame.
- `postprocess(raw, prepared) -> FrameTracks`: organize results in input detection order.

`predict` calls these stages in sequence. Use one instance per video stream and process frames in timestamp order.
Call `reset()` before starting a new video or replaying from the beginning; track IDs restart at 1.

<a id="troubleshooting"></a>
## Troubleshooting

- `libhobot_mot_capi.so` not found: run the build commands above or select an existing library with `--native-lib`.
- Invalid `use_kalman_filter` type: use JSON `true` or `false`.
- Timestamp not increasing: check frame order; call `reset()` before replaying from the beginning.
- Configuration file changed: create a new `HobotMotTracker` before running.

