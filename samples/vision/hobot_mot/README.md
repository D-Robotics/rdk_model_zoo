English | [简体中文](README_cn.md)

<a id="overview"></a>
## HobotMot (IOU 2.0) multi-object tracker

HobotMot is a CPU multi-object tracker from D-Robotics. Supply bounding boxes, confidence scores and classes for each frame to obtain track IDs and notifications when tracks disappear. Connect a detector such as YOLO by converting its results to the input format shown below.

- Upstream source: <https://github.com/D-Robotics/mot> (Apache-2.0), pinned
  to commit `0120205146d88c3474bfc2936522e349e169fa8d`
- Algorithm: IOU 2.0 (`tracker_type: "IOU_2.0"`), optionally with a Kalman
  filter for track smoothing (`use_kalman_filter: true`)
- Timestamps are milliseconds and must strictly increase; output rows keep
  the input order (one row per input detection)

<a id="directory"></a>
## Directory structure

```text
hobot_mot/
├── model/               # pinned upstream source preparation + tracker configs
├── runtime/python/      # Python replay CLI and HobotMotTracker model class
├── runtime/cpp/         # C++ tracker, C ABI binding and replay binary
└── test_data/           # 72-frame detection example
```

<a id="support-matrix"></a>
## Supported boards

| Board | Python | C++ |
| --- | --- | --- |
| RDK X5 | yes | yes |
| RDK S100 | yes | yes |
| RDK S100P | yes | yes |
| RDK S600 | yes | yes |

Use the same source and configuration on all four boards, and build on each board. Both Python and C++ call the native HobotMot library.

<a id="prerequisites"></a>
## Prerequisites

- RDK X5, S100, S100P or S600 (ARM64 Linux)
- C++17 compiler, CMake 3.16+ and Python 3.8+
- About 20 MB of available disk space
- Network access for the source download; see [model preparation](model/README.md) for offline setup

<a id="quickstart"></a>
## Quick start

From this sample directory on a board:

```bash
python3 model/download.py
cmake -B runtime/cpp/build -S runtime/cpp && cmake --build runtime/cpp/build
./runtime/cpp/run.sh
```

`download.py` fetches the pinned upstream source, verifies its SHA-256 and
extracts it under `model/vendor/`. The build compiles the tracker and the
`hobot_mot_replay` binary. `run.sh` replays `test_data/detections.jsonl`
through the default config (`model/configs/iou2.json`) and prints one line
per frame; a successful run exits with code 0.

<a id="expected-results"></a>
## Expected results

With the included example sequence the first frames print (C++ runtime):

```text
frame 0 ts 1000: [cls 0 id 1 (100,100,160,260) score 0.95 valid] [cls 0 id 2 (400,300,460,460) score 0.92 valid] [cls 1 id 1 (480,200,600,300) score 0.9 valid]
frame 1 ts 1040: [cls 0 id 1 (104,100,164,260) score 0.95 valid] [cls 0 id 2 (400,300,460,460) score 0.92 valid] [cls 1 id 1 (474,200,594,300) score 0.9 valid]
...
frame 10 ts 1400:
...
frame 70 ts 3800: [cls 0 id 1 (380,100,440,260) score 0.95 valid] [cls 0 id 2 (400,300,460,460) score 0.92 valid] [removed cls 1 id 1]
```

Reading the output: each input detection gets one row (same order), the
walking person holds `id 1`, the static person holds `id 2`, frame 10 is an
empty input frame (no rows, tracks age), the low-score decoy on frames 12–14
stays `id -1 invalid` (below `min_score`), and the car that leaves after
frame 19 is reported `[removed cls 1 id 1]` once its disappearance window
(`vanish_frame_count * time_gap` = 50 × 40 ms) expires. Pass `--output
results.json` to also write the full result JSON (schema in
[runtime/cpp/README.md](runtime/cpp/README.md)).

<a id="entry-points"></a>
## Entry points

| Entry | Purpose |
| --- | --- |
| [model/download.py](model/download.py) | fetch + verify the pinned upstream source (see [model/README.md](model/README.md)) |
| [runtime/python/main.py](runtime/python/main.py) | Python replay CLI and `HobotMotTracker` class (see [runtime/python/README.md](runtime/python/README.md)) |
| [runtime/cpp/run.sh](runtime/cpp/run.sh) | run the C++ replay binary (see [runtime/cpp/README.md](runtime/cpp/README.md)) |

<a id="license"></a>
## License

The upstream D-Robotics/mot sources are Apache License 2.0 (copyright
D-Robotics); see [runtime/cpp/THIRD_PARTY_LICENSES.md](runtime/cpp/THIRD_PARTY_LICENSES.md).
The sample code in this directory follows the repository license.
