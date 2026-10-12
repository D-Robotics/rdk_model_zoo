English | [简体中文](README_cn.md)

# ByteTrack C++ Runtime

<a id="overview"></a>
## C++ inference

Use this directory for C++ inference. YOLOv5 detects people on the board BPU; the C++ BYTETracker associates detections into tracks on the CPU. The `bytetrack` executable offers two modes: `video` (board builds) reads a prepared video plus a detector model and writes tracks, an optional annotated video and an optional records TSV; `replay` (host and board builds) feeds a detection sequence from a file to the tracker alone. Option names mirror the synonymous Python CLI options (`--input`, `--model-path`, `--output`, `--records`, thresholds, scheduling).

Video mode uses the YOLOv5 C++ detector's letterbox preprocessing and floating-point coordinates. The Python X5 detector uses direct resizing, clips coordinates and converts them to integers, so the two video modes can produce different tracks; see [Interpreting Results](#results-interpretation). Applications with an existing detector can call `ByteTracker` directly and supply their own detections.

<a id="directory"></a>
## Directory structure

```text
cpp/
├── CMakeLists.txt  # Explicit-target build: host, x5, s100, s100p, s600
├── run.sh  # Build (once) and run the executable for one target
├── inc/
│   ├── tracking.hpp  # ByteTracker + ByteTrackModel public API (SDK-free)
│   ├── cli.hpp  # CLI options, replay reader, records writer, video reader/sink
│   ├── parallel.hpp  # Fixed-slot runner for per-frame parallelism
│   └── lapjv.h  # Vendored linear-assignment solver interface
├── src/
│   ├── tracking.cpp  # BYTETracker port + board model composition over YOLOv5
│   ├── lapjv.cpp  # Vendored lap 0.5.12 solver (verbatim)
│   ├── parallel.cpp  # Thread pool behind ByteTracker's parallel regions
│   ├── cli.cpp  # Argument parsing, replay IO, records TSV, video IO/rendering
│   └── main.cpp  # Thin entry: parse -> construct model -> predict per frame
├── THIRD_PARTY_LICENSES.md
└── licenses/  # lap-LICENSE
```

`inc/tracking.hpp` and `inc/cli.hpp` include no board SDK headers: the SDK types stay inside the `.cpp` translation units, so any application can include the public headers and link the library.

<a id="supported-boards"></a>
## Supported Boards

| Board | C++ entry | Requirements |
| --- | --- | --- |
| X5 | `bytetrack video` / `replay` | OpenCV, system HB-DNN SDK, X5 YOLOv5 640×640 packed-NV12 `.bin` model (default `n-v7.0`; see [model guide](../../model/README.md)) |
| S100 | `bytetrack video` / `replay` | OpenCV, system DNN/UCP SDK, `s100/yolov5x_672x672_nv12.hbm` |
| S100P | `bytetrack video` / `replay` | OpenCV, system DNN/UCP SDK, `s100p/yolov5x_672x672_nv12.hbm` |
| S600 | `bytetrack video` / `replay` | OpenCV, system DNN/UCP SDK, `s600/yolov5x_672x672_nv12.hbm` |
| Host (no board) | `bytetrack replay` | C++17 compiler; no SDK and no OpenCV needed |

The S-target artifacts are target-specific: use each target's own `.hbm` row from the [model guide](../../model/README.md).

<a id="dependencies"></a>
## Dependencies

- Toolchain: a C++17 compiler — either the board image's native GCC (measured: GCC 11.2 on X5) or an OE cross toolchain for the target.
- CMake ≥ 3.16 with `Threads`.
- Board builds: OpenCV development libraries (video decode/encode and rendering).
- X5: system HB-DNN SDK (`libdnn`; headers under `/usr/hobot/include` and `/usr/include/hobot/dnn`).
- S100/S100P/S600: system DNN/UCP SDK (`libdnn`, `libhbucp`; headers under `/usr/hobot/include` and `/usr/include/hobot`) plus this repository's `utils/c_utils` sources, which the build adds automatically.
- SDK location overrides (cache variables): `BYTETRACK_HOBOT_LIB_DIR` (default `/usr/hobot/lib`), `BYTETRACK_HOBOT_INCLUDE_DIR` (default `/usr/hobot/include`), `BYTETRACK_SYSTEM_HOBOT_INCLUDE_DIR` (default `/usr/include/hobot`).

<a id="build"></a>
## Build

The target is explicit; CMake does not probe the SoC; nothing is auto-detected. `BYTETRACK_TARGET` must be one of `host`, `x5`, `s100`, `s100p`, `s600`, and each target builds in its own directory so switching targets never reuses another target's objects:

```bash
# cwd: samples/vision/bytetrack/runtime/cpp
cmake -S . -B build/x5 -DBYTETRACK_TARGET=x5 -DCMAKE_BUILD_TYPE=Release
cmake --build build/x5 --target bytetrack -j
# expect: build/x5/bytetrack
```

Use `-DBYTETRACK_TARGET=s100` (or `s100p`, `s600`, `host`) the same way; the binary lands in `build/<target>/bytetrack`.

The launcher wraps the same two commands (run from the repository root):

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- --help
```

<a id="run"></a>
## Run

Prepare the model and video first (the executable never downloads anything): follow the [model guide](../../model/README.md) and [Sample quickstart](../../README.md#quickstart). Then, from the repository root:

```bash
# X5: video tracking with the default detector variant (n-v7.0)
bash samples/vision/bytetrack/runtime/cpp/run.sh --target x5 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --output result_x5.mp4 --records tracks_x5.tsv --threads 1
# expect: "video: <N> frames, <M> track rows", result_x5.mp4, tracks_x5.tsv
```

S targets pass their own model row and target name:

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target s100 -- video \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --output result_s100.mp4 --records tracks_s100.tsv
```

Substitute `s100p`/`s600` and the matching `model/<target>/yolov5x_672x672_nv12.hbm` for the other S boards. Omit `--output` to disable rendering and video encoding; omit `--records` to disable the TSV; both writers are independent.

The host target needs no board SDK and runs the tracker alone from a detection file:

```bash
bash samples/vision/bytetrack/runtime/cpp/run.sh --target host -- replay \
  --detections persons.btcpp --records tracks.tsv --threads 1
# expect: "replay: <N> frames, <M> track rows, final frame_id <F>"
```

<a id="parameters"></a>
## Parameters

`bytetrack <mode> [options]` with mode `replay` or `video` (`video` exists on board builds only; a host binary refuses it). Defaults are the values the code defines:

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
| `--mot20` | off | source optional matching mode (no score fusion) |
| `--bpu-cores I` | -1 | BPU core: -1 any, or one fixed index 0..3 |
| `--priority I` | 0 | BPU task priority, 0..255 |
| `--max-frames N` | 0 | stop after N frames; 0 processes the whole input |
| `--threads N` | 1 | tracker-internal parallelism, including the caller |

On X5, keep `--priority 0` and `--bpu-cores -1`. S targets accept a single core index with `--bpu-cores`; the Python CLI accepts a list.

An output path (`--records` or `--output`) that names an input (`--detections`, `--input`, `--model-path`) — the same file, a symlink alias, or an equivalent not-yet-existing path — is rejected before anything is opened. `--records` and `--output` must name different files.

<a id="interface-lifecycle"></a>
## Interface & Resource Lifecycle

`inc/tracking.hpp` exposes two interfaces; both are RAII — resources are allocated in the constructor and released in the destructor, there is no `init()`/`deinit()` pair.

- Config: `TrackingConfig{track_thresh=.3, track_buffer=60, match_thresh=.8, frame_rate=30, mot20=false}` — `frame_rate` scales `track_buffer/30`; out-of-range values throw `TrackingConfigError` from the constructor.
- Model: `ByteTracker(config, threads)` validates the config and builds tracker state; `update(detections, img_h, img_w)` advances one frame and returns owned `Track` snapshots (`track_id`, `tlbr` XYXY doubles in original pixels, `score`, `frame_id`); `frame_id()`, `persistent_track_count()`, `parallel_usage()`, `reset()`. `update` throws `TrackingInputError` for NaN coordinates/scores or non-positive boxes/sizes.
- Model: `ByteTrackModel::Options{model_path, score_thres=.25, nms_thres=.45, bpu_priority=0, bpu_core=-1, tracking, threads=1}` — board builds only. The constructor loads the YOLOv5 model and allocates BPU tensors (throwing on failure); the destructor releases them.
- Data flow: `preprocess(const Frame&) -> Prepared` (letterbox/tensor preparation), `infer(const Prepared&) -> Raw` (BPU forward), `postprocess(const Raw&, const Prepared&) -> std::vector<Track>` (decode persons, COCO class 0, one tracker update); `predict(const Frame&)` composes the three stages exactly once. `detect(const Frame&)` returns the detection list without advancing the tracker. `Frame` owns HWC BGR bytes; `Prepared`/`Raw` own the detector payloads passed by const reference.
- Threading: one instance per ordered video stream; instances are not thread safe. `--threads` parallelizes per-track Kalman prediction and IoU rows inside a frame (results identical for any count); association stays sequential. For multiple streams, give each stream its own instance and thread.
- Video IO (`inc/cli.hpp`): `VideoReader` opens the input in its constructor; `VideoSink` opens the output lazily on the first `write` (mp4v, the frame's size) and throws when the path cannot be written; both release OpenCV handles in their destructors.

Use one instance per stream and process frames in order, including frames with empty detections. `reset()` clears track history and the frame index while retaining the increasing ID counter. Each instance has its own ID domain; use `(stream_id, track_id)` in applications with multiple streams.

<a id="results-interpretation"></a>
## Interpreting Results

On success the program prints one summary line — `video: N frames, M track rows` or `replay: N frames, M track rows, final frame_id F` — and exits `0`. Exit `2` marks a command-line error (message plus usage); exit `1` marks a runtime error (model load, video open, records write). Failed record writes are reported, never silently dropped.

`--records` writes a TSV with header `frame lane track_id x1 y1 x2 y2 score`: frames are 1-based, the CLI uses lane 0, coordinates are original-image pixels at `%.17g`. Frames without output tracks have no TSV rows but still advance tracking state. `--output` writes an MP4 with track boxes and `ID:<n>` labels using the same color palette as the Python renderer.

The replay input is little-endian binary: an 8-byte magic `BTCPP1\0\0`, four `uint32` fields (version 1, frame count, image height, image width), then per frame a `uint32` detection count and five `float32` values per detection (x1, y1, x2, y2, score). Coordinates must be finite and boxes must have positive area.

Current behavior, full-video detection path: C++ video mode uses the YOLOv5 C++ detector's letterbox preprocessing (linear resize, gray-127 padding) and keeps floating-point, unclamped coordinates, while the Python X5 path uses direct nearest-neighbor resizing, clipping and integer truncation. These preprocessing and coordinate conventions can produce different detections and tracks for the same video. To use a shared detector, pass its detections directly to `ByteTracker::update`.

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

The one-core budget is CPU milliseconds/frame × FPS ÷ 10. Single-thread process RSS is about 8.5–12.4 MiB, including shared libraries, input sequences and timing records. The 100-person tier starts at about 11.1 MiB after loading inputs; total RSS is not memory used exclusively by tracking state.

For the 100-person input, internal threads and independent streams perform as follows:

| Configuration | Mean latency per stream (ms/frame) | CPU time per stream-frame (ms) | Aggregate stream-frames/s |
|---|---:|---:|---:|
| 1 stream, 1 thread | 1.912 | 1.912 | 522 |
| 1 stream, 2 threads | 1.897 | 1.962 | 526 |
| 1 stream, 4 threads | 1.850 | 1.963 | 539 |
| 1 stream, 8 threads | 1.915 | 1.912 | 521 |
| 2 streams, 1 thread each | 1.955 | 1.950 | 1020 |
| 4 streams, 1 thread each | 2.140 | 2.134 | 1864 |

CPU time is measured once across the process window and divided by the total stream-frame count. Aggregate throughput is not the frame rate of one stream. The 100-person, 8-thread configuration remains below the parallel chunk threshold and executes serially. Four internal threads reduce single-stream latency by only about 3%, while four independent streams deliver about 3.57 times the aggregate throughput of one stream. Keep the default `--threads 1` and parallelize independent streams first. Four-stream process RSS is about 15.7 MiB.

CPU frequency scaling uses the system defaults; short runs at small person counts are sensitive to frequency changes and scheduling. These are tracking-only throughput figures. Full video speed also depends on detection and video input/output.
