English | [简体中文](README_cn.md)

<a id="overview"></a>
## C++ runtime

Track detection sequences with `HobotMotTracker`, or read a JSONL file with `hobot_mot_replay`.
The build also produces `libhobot_mot_capi.so` for the Python runtime.

<a id="directory"></a>
## Directory structure

```text
runtime/cpp/
├── inc/
│   ├── track.hpp       # HobotMotTracker: preprocess/infer/postprocess/reset
│   ├── mot_c_api.h     # C ABI of libhobot_mot_capi.so (ctypes-compatible)
│   └── cli.hpp         # replay CLI helpers
├── src/
│   ├── track.cpp       # config validation + per-class upstream routing
│   ├── mot_c_api.cpp   # C boundary: error codes, handle ownership
│   ├── cli.cpp         # JSONL loading, rendering, result JSON
│   └── main.cpp        # visible entry point
├── CMakeLists.txt      # builds hobot_mot_capi + hobot_mot_replay
├── run.sh              # runs build/hobot_mot_replay
├── THIRD_PARTY_LICENSES.md
└── build/              # CMake build output
```

<a id="supported-boards"></a>
## Supported boards

Supports RDK X5, S100, S100P and S600. Use the following commands to build and run on each board.

<a id="dependencies"></a>
## Dependencies

- CMake >= 3.16 and a C++17 compiler on the board
- The prepared upstream tree at
  `model/vendor/mot-0120205146d88c3474bfc2936522e349e169fa8d/` (override with
  `-DHOBOT_MOT_SOURCE_ROOT=<path>`)
- Linked from that tree: the prebuilt ARM64 `iou_based_mot` core plus
  `hobotlog`, `hobotsdk`, `hobotthread` and `jsoncpp` archives; system
  `rt`, `dl`, `pthread`. No OpenCV, no BPU runtime
- Compile definitions: `-DHR_POSIX -DENABLE_IOU2` (the upstream build flags
  for the IOU2-only configuration)

<a id="build"></a>
## Build

From the sample root:

```bash
python3 model/download.py
cmake -B runtime/cpp/build -S runtime/cpp
cmake --build runtime/cpp/build
```

Outputs: `runtime/cpp/build/hobot_mot_replay` and `runtime/cpp/build/libhobot_mot_capi.so`.

<a id="run"></a>
## Run

```bash
./runtime/cpp/run.sh                                  # defaults
./runtime/cpp/run.sh --config model/configs/iou2_kalman.json --output results.json
```

Defaults: input `test_data/detections.jsonl`, config `model/configs/iou2.json`,
frame size 960x544. Exit code 0 on success, 2 with `Error: ...` on stderr
for invalid input or config.

<a id="parameters"></a>
## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--input` | path | `test_data/detections.jsonl` | detection JSONL |
| `--config` | path | `model/configs/iou2.json` | tracker config JSON |
| `--output` | path | none | also write full results as JSON |
| `--frame-width` | int | 960 | frame width when a record has no `image_width` |
| `--frame-height` | int | 544 | frame height when a record has no `image_height` |
| `--help` | flag | — | usage text (exit 0) |

<a id="interface-lifecycle"></a>
## Interface and call order

Create `HobotMotTracker` with `TrackOptions{config_path, frame_width, frame_height}`.
Use one instance per video stream and call `predict(detections, timestamp_ms, width, height)` in frame order to get `FrameTracks`.
For a frame with no detections, pass an empty list and its timestamp.

You can also call `preprocess`, `infer` and `postprocess` in sequence to inspect each stage.
Call `reset()` before starting a new video or replaying from the beginning; track IDs restart at 1.
Create a new instance after changing the configuration. Configuration errors raise `MotConfigError`; input errors raise `MotInputError`.

The C interface used by Python is defined in [inc/mot_c_api.h](inc/mot_c_api.h).

<a id="results-interpretation"></a>
## Results interpretation

One line per frame on stdout:

```text
frame <frame_index> ts <timestamp_ms>: [cls <class_id> id <track_id> (<x1>,<y1>,<x2>,<y2>) score <score> <state>] ... [removed cls <class_id> id <track_id>] ...
```

Rows keep the input order (one row per input detection). `state valid` means
matched with identity `track_id` (per-class namespaces, both starting at 1);
`invalid` rows carry `track_id` -1. `[removed ...]` lists identities dropped
this frame after the disappearance window. `--output` writes the same data
as JSON (schema in [../python/README.md](../python/README.md)). Exit code 2
plus `Error: ...` on stderr marks rejected input or config; no partial
result file is written in that case.
