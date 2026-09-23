English | [简体中文](./README_cn.md)

# ByteTrack

<a id="overview"></a>
## Algorithm and source

ByteTrack is a stateful multi-object tracker that associates high- and low-score detections. This sample runs an S100/S100P/S600 YOLOv5x detector, keeps COCO `person` class `0`, and updates the CPU BYTETracker. The source paper is [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864).

<a id="support-matrix"></a>
## Support and verification matrix

| target | variant | Python | C++ | status |
|---|---|---|---|---|
| S100 | YOLOv5x 672 | supported-not-run | not-supported | host tracker fixtures; board not-run |
| S100P | YOLOv5x 672 | supported-not-run | not-supported | host tracker fixtures; board not-run |
| S600 | YOLOv5x 672 | supported-not-run | not-supported | host tracker fixtures; board not-run |
| X5 | — | not-supported | not-supported | no ByteTrack asset |

The tracker is stateful: one `ByteTrackTask` must process frames in order. `reset()` clears stream history and frame index but deliberately keeps the process-global track ID counter monotonic.

<a id="prerequisites"></a>
## Prerequisites

Host tracker checks use Python, NumPy, SciPy, OpenCV, `lap==0.5.12`, and `cython-bbox==0.1.5`. Board inference additionally needs the target `hbm_runtime` and the exact target HBM. The source video is missing from `test_data`; prepare it explicitly from `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4`. No model/video download was performed here.

<a id="quickstart"></a>
## Quick start

From the repository root, explicitly prepare S100 model and video, then run:

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

The first two commands are explicit preparation commands and were not run. Successful inference writes a decodable MP4 and optional per-frame JSONL, prints the frame count, and exits `0`. `run.sh` never installs or downloads them.

<a id="expected-results"></a>
## Expected results

Each processed frame yields zero or more person tracks with `track_id`, original-image `tlbr`, score, and frame index. A result video is written at the requested output path. Empty detections still advance `frame_index` and update the tracker. A source edge case exists when a box lies wholly in letterbox padding: clipping can create zero area and source XYAH initialization may produce NaN; the unified task drops non-positive-width/height person boxes before tracker update. The evaluator records any source NaN as an error and fails rather than treating it as equality.

<a id="directory"></a>
## Directory

```text
.
├── model/                 # explicit S HBM preparation
├── runtime/python/        # detector binding, tracker state, CLI, source map
├── conversion/            # detector conversion boundary and OE resources
├── evaluator/             # fresh-process full capture/comparison
├── test_data/              # images, labels, reference GIFs; video is external
└── tests/                 # CPU tracker, source comparison, CLI and evidence fixtures
```

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): three target HBM rows and explicit preparation.
- [`runtime/python/README.md`](./runtime/python/README.md): stateful task API and full CLI.
- [`conversion/README.md`](./conversion/README.md): detector-only conversion boundary.
- [`evaluator/README.md`](./evaluator/README.md): fresh-process source/unified evidence and strict IDs.

<a id="historical-performance"></a>
## Historical source performance

The source records tracker update time about `2.37 ms` on RDK S100. The source paper reports `80.3 MOTA`, `77.3 IDF1`, and about `30 FPS` on a V100 GPU. These are historical references, not this migration's board measurements.

<a id="license"></a>
## License

Repository wrappers follow Apache-2.0. The tracker source tree has no separate license file in the fixed source inventory; its provenance and `TRACKER_SOURCE_MAP.json` are retained. Upstream ByteTrack and any model weights keep their own licenses.
