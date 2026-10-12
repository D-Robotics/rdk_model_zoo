English | [简体中文](README_cn.md)

# ByteTrack

<a id="overview"></a>
## Algorithm and source

ByteTrack tracks objects across video frames by associating both high- and low-confidence detections. Recovering low-score detections helps maintain identities when pedestrians are partly occluded. This sample combines a YOLOv5 person detector with the CPU BYTETracker.

References: [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864).

<a id="directory"></a>
## Directory structure

```text
bytetrack/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

| target | variant | Python | C++ | note |
|---|---|---|---|---|
| S100 | YOLOv5x 672 | supported | supported | Target-specific HBM |
| S100P | YOLOv5x 672 | supported | supported | Target-specific HBM |
| S600 | YOLOv5x 672 | supported | supported | Target-specific HBM |
| X5 | YOLOv5 640, default n-v7.0 | supported | supported | Uses X5 YOLOv5 `.bin` models |

Use one `ByteTrackTask` instance per video and process frames in order. When switching videos, call `reset()` to clear existing tracks and restart the frame count. New track IDs continue to increase.

<a id="prerequisites"></a>
## Prerequisites

Use Python 3, NumPy, SciPy, OpenCV, `lap==0.5.12`, and `cython-bbox==0.1.5`. X5 uses the image-provided `hbm_runtime` and 640×640 packed-NV12 `.bin` models. S targets use `hbm_runtime` and 672×672 split-NV12 `.hbm` models. Install the CPU dependencies:

```bash
python3 -m pip install -r samples/vision/bytetrack/requirements-host.txt
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 model and input video, then run:

```bash
python3 -m samples.vision.bytetrack.model.download --target x5 --variant n-v7.0
curl -fL 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4' \
  -o samples/vision/bytetrack/test_data/track_test.mp4
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target x5 --variant n-v7.0 \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_x5.mp4 \
  --records samples/vision/bytetrack/test_data/result_x5.jsonl
```

`--variant` supports `n-v7.0`, `s-v7.0`, `m-v7.0`, `l-v7.0`, `x-v7.0`, and `s-v2.0`, `m-v2.0`, `l-v2.0`, `x-v2.0`. Use the same variant for download and inference.

For S100:

```bash
python3 -m samples.vision.bytetrack.model.download --target s100 \
  --output-dir samples/vision/bytetrack/model
curl -L 'https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4' \
  -o samples/vision/bytetrack/test_data/track_test.mp4
python3 -m samples.vision.bytetrack.runtime.python.main \
  --target s100 --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output samples/vision/bytetrack/test_data/result_unified.mp4 \
  --records samples/vision/bytetrack/test_data/result_unified.jsonl
```

The recorded SHA-256 of the published `track_test.mp4` is `4bbe5bf11fe8967b28a900fd2add4949aba89b62076eaa03d0c55cdf7dd41397`; verify the download with `sha256sum track_test.mp4` (macOS: `shasum -a 256 track_test.mp4`) when integrity matters. Successful inference writes a decodable MP4 and optional per-frame JSONL, prints the frame count, and exits `0`.

<a id="expected-results"></a>
## Expected results

The output video shows person boxes and track IDs. Each frame record contains the track ID (`track_id`), original-image coordinates (`tlbr`), confidence (`score`), and frame number. The tracker updates existing tracks even when no people are detected.

Lower `--score-thres` if people are missed. Increase `--track-buffer` to retain tracks through longer occlusions. See the [Python parameters](runtime/python/README.md#parameters) for the full option list.

The detector keeps COCO `person` class `0`. BYTETracker first matches high-score boxes, then associates remaining tracks with low-score boxes using IoU; only unmatched high-score detections initialize new tracks.

![ByteTrack detection example strip](test_data/readme_img/image1.png)
![Three-row (a)/(b)/(c) association illustration](test_data/readme_img/image.png)

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): X5 BIN and S-target HBM preparation.
- [`runtime/python/README.md`](./runtime/python/README.md): Python API and command-line options.
- [`runtime/cpp/README.md`](./runtime/cpp/README.md): C++ video tracking on X5/S100/S100P/S600, host detections replay, CPU tracking API and thread configuration.
- [`conversion/README.md`](./conversion/README.md): YOLOv5 export and compilation.
- [`evaluator/README.md`](./evaluator/README.md): evaluation commands, outputs, and comparison notes.

<a id="historical-performance"></a>
## Performance and resource use

For crowded scenes on X5, the [C++ runtime](runtime/cpp/README.md#x5-cpu-performance) averages about 1.91 ms per frame with one thread at the 100-person tier, using about 5.7% of one CPU core at 30 FPS. Keep the default single thread and parallelize independent video streams first. Detection runs on the BPU; track prediction and association run on the CPU. The Python figures below help with choosing a runtime.

### Tracking computation

The following board measurements, recorded on 2026-10-12, cover only `BYTETracker.update`. Both boards use the same synthetic motion sequences, including low-score detections, occlusion and empty frames. Each tier has 100 warmup frames followed by 1200 recorded frames, repeated twice. The person count is the generated maximum; active tracks are the mean output count. Latency statistics combine both repeats.

| Board | Maximum people | Mean active tracks | Mean / P95 latency (ms) | CPU time (ms/frame) | Single-core budget at 30 FPS | Single-core budget at 60 FPS |
|---|---:|---:|---:|---:|---:|---:|
| X5 | 5 | 4.6 | 4.46 / 5.29 | 4.43 | 13.3% | 26.6% |
| X5 | 20 | 18.2 | 12.91 / 14.50 | 12.86 | 38.6% | 77.2% |
| X5 | 50 | 44.6 | 30.12 / 33.93 | 30.02 | 90.1% | 180.1% |
| X5 | 100 | 90.7 | 60.51 / 68.04 | 60.34 | 181.0% | 362.0% |
| S100 | 5 | 4.6 | 1.47 / 1.72 | 1.46 | 4.4% | 8.7% |
| S100 | 20 | 18.2 | 3.60 / 4.00 | 3.57 | 10.7% | 21.4% |
| S100 | 50 | 44.6 | 7.63 / 8.64 | 7.58 | 22.7% | 45.5% |
| S100 | 100 | 90.7 | 14.71 / 16.45 | 14.63 | 43.9% | 87.8% |

Single-core budget = CPU milliseconds/frame × FPS ÷ 10. A value of 100% uses one full CPU core; values above 100% exceed the frame rate this configuration can sustain. Divide by 8 on X5 or 6 on S100 to express use as a share of all CPU cores.

Environment: Ubuntu 22.04, Python 3.10.12, NumPy 1.26.4, lap 0.5.13, cython-bbox 0.1.5, and one BLAS/OpenMP thread. X5 uses SciPy 1.10.1; S100 uses 1.15.3. Both CPUs have a 1.5 GHz frequency limit and use system frequency scaling. Some S100 samples ran at 1.125 GHz, with background load present.

Tracking-process memory (RSS) is about 49–63 MiB on X5 and 69–85 MiB on S100. This includes the Python interpreter, libraries, input sequences, and timing records. Initial use after loading these items is 46.8 MiB and 68.9 MiB, respectively. Person-count tiers run sequentially in the same process.

### Video pipeline

The input is this Sample's `track_test.mp4`, at 1280×720 and 30 FPS. Each of two repeats processes 300 recorded frames after 30 warmup frames. The table averages both repeats, using default detection and tracking thresholds and OpenCV `mp4v` output.

| Stage | X5: YOLOv5n v7.0 640 (ms/frame) | S100: YOLOv5x 672 (ms/frame) |
|---|---:|---:|
| Video decode | 7.99 | 2.92 |
| Detection preprocessing | 2.52 | 2.59 |
| Model Runtime call | 22.28 | 24.05 |
| Detection postprocessing | 19.55 | 36.78 |
| Person filtering and tracker update | 6.60 | 3.01 |
| Drawing | 2.50 | 1.36 |
| Video encoding and writing | 70.56 | 21.31 |
| Total | 132.02 | 92.03 |

X5 and S100 average 5.72 and 10.94 active tracks, using the detectors named in the table. Processing speed including video output is about 7.5 FPS and 10.8 FPS. Process CPU use is about 125% and 104% of one core, or 15.7% and 17.4% across all cores. CPU use includes encoder shutdown; stage timings cover frame processing. Peak RSS during frame processing is 146.9 MiB and 163.9 MiB; whole-process peaks are 146.9 MiB and 191.6 MiB. Memory figures include detection, tracking, and video I/O.

X5 uses HBRT 3.15.55.0 with a model built for 3.15.54.0. S100 uses UCP 3.13.6. Model and SDK versions affect runtime behavior.

### CPU and BPU responsibilities

This sample runs YOLOv5 detection on the BPU and track prediction and association on the CPU.

Use the [C++ runtime](runtime/cpp/README.md) for crowded scenes or multiple video streams. Create one tracker per stream and start with `--threads 1`. If you only need track records, omit `--output` from the C++ command to skip drawing and video encoding.

<a id="license"></a>
## License

Repository scripts use Apache-2.0. ByteTrack source and model weights retain their upstream licenses. See [`runtime/python/TRACKER_SOURCE_MAP.json`](./runtime/python/TRACKER_SOURCE_MAP.json) for tracker source attribution.
