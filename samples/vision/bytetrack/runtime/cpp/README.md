English | [简体中文](README_cn.md)

# ByteTrack C++ Runtime

<a id="overview"></a>
## C++ inference

YOLOv5 detects people on the BPU, and ByteTrack associates tracks on the CPU. Run `bytetrack video` to process video and save annotated video or track records. Run `bytetrack replay` to track objects from existing detection results.

If your application already has a detector, pass its results to `ByteTracker::update`. See [Interpreting Results](#results-interpretation) for video preprocessing and coordinate formats.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── CMakeLists.txt  # Explicit-target build: host, x5, s100, s100p, s600
├── run.sh  # Build and run for one target
├── inc/
│   ├── tracking.hpp  # ByteTracker + ByteTrackModel public API (SDK-free)
│   ├── cli.hpp  # CLI options, replay reader, records writer, video reader/sink
│   ├── parallel.hpp  # Thread scheduling within a frame
│   └── lapjv.h  # Vendored linear-assignment solver interface
├── src/
│   ├── tracking.cpp  # Tracking and YOLOv5 inference
│   ├── lapjv.cpp  # Vendored lap 0.5.12 solver (verbatim)
│   ├── parallel.cpp  # Thread pool behind ByteTracker's parallel regions
│   ├── cli.cpp  # Argument parsing, replay IO, records TSV, video IO/rendering
│   └── main.cpp  # Entry: parse -> construct model -> predict per frame
├── THIRD_PARTY_LICENSES.md
└── licenses/  # lap-LICENSE
```

To use the tracker in a C++ application, include `inc/tracking.hpp` and link `bytetrack_core`. Video I/O interfaces are in `inc/cli.hpp`.

<a id="supported-boards"></a>
## Supported Boards

| Board | C++ entry | Requirements |
| --- | --- | --- |
| X5 | `bytetrack video` / `replay` | OpenCV, system HB-DNN SDK, X5 YOLOv5 640×640 packed-NV12 `.bin` model (default `n-v7.0`; see [model guide](../../model/README.md)) |
| S100 | `bytetrack video` / `replay` | OpenCV, system DNN/UCP SDK, `s100/yolov5x_672x672_nv12.hbm` |
| S100P | `bytetrack video` / `replay` | OpenCV, system DNN/UCP SDK, `s100p/yolov5x_672x672_nv12.hbm` |
| S600 | `bytetrack video` / `replay` | OpenCV, system DNN/UCP SDK, `s600/yolov5x_672x672_nv12.hbm` |
| Host (no board) | `bytetrack replay` | C++17 compiler; no SDK and no OpenCV needed |

Select the model for your board from the [model guide](../../model/README.md).

<a id="dependencies"></a>
## Dependencies

- Toolchain: a C++17 compiler, using the board’s GCC or an OE cross compiler for your target.
- CMake ≥ 3.16 with `Threads`.
- Board builds: OpenCV development libraries (video decode/encode and rendering).
- X5: system HB-DNN SDK (`libdnn`; headers under `/usr/hobot/include` and `/usr/include/hobot/dnn`).
- S100/S100P/S600: system DNN/UCP SDK (`libdnn`, `libhbucp`; headers under `/usr/hobot/include` and `/usr/include/hobot`) plus this repository's `utils/c_utils` sources, which the build adds automatically.
- SDK location overrides (cache variables): `BYTETRACK_HOBOT_LIB_DIR` (default `/usr/hobot/lib`), `BYTETRACK_HOBOT_INCLUDE_DIR` (default `/usr/hobot/include`), `BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR` (default `/usr/include/hobot`).

<a id="build"></a>
## Build

Set `BYTETRACK_TARGET` to `host`, `x5`, `s100`, `s100p`, or `s600`. These commands build the X5 executable:

```bash
# cwd: samples/vision/bytetrack/runtime/cpp
cmake -S . -B build/x5 -DBYTETRACK_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/x5 --target bytetrack -j
# expect: build/x5/bytetrack
```

For another target, change both `-B build/<target>` and `-DBYTETRACK_TARGET=<target>`. The executable is written to `build/<target>/bytetrack`.

The launcher wraps the same two commands (run from the repository root):

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- --help
```

<a id="run"></a>
## Run

Prepare the model and video using the [model guide](../../model/README.md) and [quickstart](../../README.md#quickstart). Then run from the repository root:

```bash
# X5: video tracking with the default detector variant (n-v7.0)
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --output result_x5.mp4 --records tracks_x5.tsv --threads 1
# expect: "video: <N> frames, <M> track rows", result_x5.mp4, tracks_x5.tsv
```

For S100:

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target s100 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --output result_s100.mp4 --records tracks_s100.tsv
```

For S100P or S600, replace `s100` in the command with your board name and use its model file. Use `--output` to save annotated video and `--records` to save tracks; either option can be omitted.

The host target needs no board SDK and runs the tracker alone from a detection file:

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target host -- replay \
  --detections persons.btcpp --records tracks.tsv --threads 1
# expect: "replay: <N> frames, <M> track rows, final frame_id <F>"
```

<a id="parameters"></a>
## Parameters

Command format: `bytetrack <mode> [options]`. Board builds support `video` and `replay`; host builds support `replay`.

| Parameter | Default | Description |
| --- | ---: | --- |
| `--detections FILE` | required (replay) | BTCPP1 float32 detection sequence |
| `--input FILE` | required (video) | input video |
| `--model-path FILE` | required (video) | detector model (X5 `.bin` / S `.hbm`) |
| `--output FILE` | none | annotated output video (mp4v); omitted disables rendering |
| `--records FILE` | none | per-frame records TSV |
| `--score-thres F` | 0.25 | detector confidence threshold, [0,1] |
| `--nms-thres F` | 0.45 | detector NMS IoU threshold, [0,1] |
| `--track-thresh F` | 0.3 | tracking activation threshold |
| `--track-buffer N` | 60 | frames a lost track survives, scaled by `frame_rate / 30` |
| `--match-thresh F` | 0.8 | first-association cost limit |
| `--frame-rate N` | 30 | source frame rate scaling the retention window |
| `--mot20` | off | MOT20 matching mode (no score fusion) |
| `--bpu-cores I` | -1 | BPU core: -1 any, or one fixed index 0..3 |
| `--priority I` | 0 | BPU task priority, 0..255 |
| `--max-frames N` | 0 | stop after N frames; 0 processes the whole input |
| `--threads N` | 1 | tracker-internal parallelism, including the caller |

On X5, keep `--priority 0` and `--bpu-cores -1`. S targets accept a single core index with `--bpu-cores`; the Python CLI accepts a list.

Use separate paths for `--output` and `--records`. Output paths must differ from the input video, detection file, and model path.

<a id="interface-lifecycle"></a>
## Interface & Resource Lifecycle

Model objects load resources when created and release them when destroyed. Choose an interface for your input:

- Existing person detections: create `ByteTracker(config, threads)` and call `update(detections, img_h, img_w)` for each frame. Returned `Track` records contain the ID, original-image coordinates in `tlbr`, confidence, and frame number.
- Video frames: create `ByteTrackModel(options)` and call `predict(frame)` for each frame. Set the model path, detector thresholds, tracking parameters, and thread count in `options`.
- Detection only: call `ByteTrackModel::detect(frame)` without updating tracks.

`TrackingConfig` defaults are `track_thresh=.3`, `track_buffer=60`, `match_thresh=.8`, `frame_rate=30`, and `mot20=false`. Creating a tracker with out-of-range parameters throws `TrackingConfigError`. `update` throws `TrackingInputError` for NaN coordinates or scores, invalid boxes, or invalid image dimensions.

For separate stage calls, use `preprocess(frame)` → `infer(prepared)` → `postprocess(raw, prepared)`. `Frame` stores an HWC BGR image, `Prepared` stores model inputs, and `Raw` stores inference outputs. Each returned object owns its data.

`--threads` controls the threads used for Kalman prediction and IoU computation within a frame; the default is `1`. Use a separate model instance and thread for each video stream.

`VideoReader` opens the video when created. `VideoSink` opens its output on the first `write`, using that frame’s dimensions and mp4v encoding. Failure to open an input or output file throws an exception.

Process frames in video order and call `update` even for frames without detections. When switching videos, call `reset()` to clear tracks and restart the frame count. New track IDs continue to increase. Each C++ instance assigns its own IDs; use `(stream_id, track_id)` to identify tracks across streams.

<a id="results-interpretation"></a>
## Interpreting Results

The program prints `video: N frames, M track rows` or `replay: N frames, M track rows, final frame_id F` and exits with code `0`. Command-line errors return `2`; model loading, video opening, or record writing errors return `1`.

`--records` writes a TSV with header `frame lane track_id x1 y1 x2 y2 score`: frames are 1-based, the CLI uses lane 0, coordinates are original-image pixels at `%.17g`. Frames without output tracks have no TSV rows but still advance tracking state. `--output` writes an MP4 with track boxes and `ID:<n>` labels using the same color palette as the Python renderer.

The replay input is little-endian binary: an 8-byte magic `BTCPP1\0\0`, four `uint32` fields (version 1, frame count, image height, image width), then per frame a `uint32` detection count and five `float32` values per detection (x1, y1, x2, y2, score). Coordinates must be finite and boxes must have positive area.

C++ video mode preserves the image aspect ratio and pads with gray 127 (letterbox), then returns floating-point coordinates. Python X5 mode resizes directly and clips coordinates to the image before converting them to integers. These preprocessing methods can produce different detections and tracks. To share detection results, call `ByteTracker::update` directly.

<a id="x5-cpu-performance"></a>
### X5 CPU performance

These tracking measurements were collected on X5 on 2026-10-12 using the same synthetic person-motion sequences, including occlusion, low-score detections and empty frames. Each tier uses 100 warmup frames followed by 1200 measured frames, repeated twice. The Release build uses GCC 11.2 and `-O3`, without fast-math. Detection, video decoding, rendering and encoding are excluded.

| Person limit | Single-thread mean / P95 (ms/frame) | CPU time (ms/frame) | One-core budget at 30 FPS |
|---|---:|---:|---:|
| 5 | 0.041 / 0.051 | 0.042 | 0.12% |
| 20 | 0.175 / 0.221 | 0.175 | 0.52% |
| 50 | 0.629 / 0.844 | 0.630 | 1.89% |
| 100 | 1.912 / 2.656 | 1.912 | 5.73% |

On the same board and inputs, Python mean latencies are 4.359, 12.825, 29.997 and 60.227 ms/frame, respectively. Single-thread C++ is approximately 105, 73, 48 and 31.5 times faster. The Python environment uses Python 3.10.12, NumPy 1.26.4, SciPy 1.10.1, lap 0.5.13 and cython-bbox 0.1.5, with BLAS/OpenMP threads set to 1.

One-core budget = CPU milliseconds/frame × FPS ÷ 10. Single-thread process RSS is about 8.5–12.4 MiB, including libraries, input sequences, tracking state, and timing records. Initial use after loading the 100-person input is about 11.1 MiB.

For the 100-person input, internal threads and independent streams perform as follows:

| Configuration | Mean latency per stream (ms/frame) | CPU time per stream-frame (ms) | Aggregate stream-frames/s |
|---|---:|---:|---:|
| 1 stream, 1 thread | 1.912 | 1.912 | 522 |
| 1 stream, 2 threads | 1.897 | 1.962 | 526 |
| 1 stream, 4 threads | 1.850 | 1.963 | 539 |
| 1 stream, 8 threads | 1.915 | 1.912 | 521 |
| 2 streams, 1 thread each | 1.955 | 1.950 | 1020 |
| 4 streams, 1 thread each | 2.140 | 2.134 | 1864 |

CPU time is total process CPU time divided by the total frames processed. Aggregate throughput is the sum of frames processed per second across all streams. At 100 people, four threads reduce single-stream latency by about 3%; four independent streams deliver about 3.57 times the throughput of one stream. For multiple videos, keep `--threads 1` per stream and process streams in parallel. Four-stream RSS is about 15.7 MiB. The eight-thread setting still uses serial computation at this person count.

CPU frequency scaling uses the system defaults; short runs at small person counts are sensitive to frequency changes and scheduling. These are tracking-only throughput figures. Full video speed also depends on detection and video input/output.
