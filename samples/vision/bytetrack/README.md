English | [简体中文](README_cn.md)

# ByteTrack

<a id="overview"></a>
## Algorithm and source

ByteTrack tracks objects across video frames by associating both high- and low-confidence detections. Recovering low-score detections helps maintain identities when pedestrians are partly occluded. This sample combines a YOLOv5x person detector with the CPU BYTETracker.

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
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Source or data file
```

<a id="support-matrix"></a>
## Support matrix

| target | variant | Python | C++ | note |
|---|---|---|---|---|
| S100 | YOLOv5x 672 | supported | not-supported | |
| S100P | YOLOv5x 672 | supported | not-supported | see download note below |
| S600 | YOLOv5x 672 | supported | not-supported | |
| X5 | — | not-supported | not-supported | no ByteTrack asset |

The S100P manifest row exists, but its published download URL has been observed to return HTTP 404; if the downloader fails, obtain the YOLOv5x HBM manually and place it at the path shown by `--list-models`.

The tracker is stateful: one `ByteTrackTask` must process frames in order. `reset` clears stream history and frame index but deliberately keeps the process-global track ID counter monotonic.

<a id="prerequisites"></a>
## Prerequisites

Host tracker checks use Python, NumPy, SciPy, OpenCV, `lap==0.5.12`, and `cython-bbox==0.1.5`. Board inference additionally needs the target `hbm_runtime` and the exact target HBM. The test video is not bundled in `test_data`; prepare it explicitly from `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4`.

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

The recorded SHA-256 of the published `track_test.mp4` is `4bbe5bf11fe8967b28a900fd2add4949aba89b62076eaa03d0c55cdf7dd41397`; verify the download with `sha256sum track_test.mp4` (macOS: `shasum -a 256 track_test.mp4`) when integrity matters. Successful inference writes a decodable MP4 and optional per-frame JSONL, prints the frame count, and exits `0`. `run.sh` never installs or downloads them.

<a id="expected-results"></a>
## Expected results

Each processed frame yields zero or more person tracks with `track_id`, original-image `tlbr`, score, and frame index. A result video is written at the requested output path. Empty detections still advance `frame_index` and update the tracker. When a box lies wholly in letterbox padding, clipping can create zero area and XYAH initialization may produce NaN; the task drops non-positive-width/height person boxes before the tracker update.

Applicability and tuning. `--score-thres` (default `0.25`) filters detector boxes before the tracker: lower it when too few boxes are detected — lowering `--track-thresh` cannot restore detector-discarded boxes. `--track-thresh` (`0.3`) partitions tracker input only: scores above it enter first association, scores in (0.1, `track-thresh`) enter second association with still-tracked targets, and new tracks start only from first-association boxes scoring at least `track_thresh + 0.1`. If track IDs switch frequently, a larger `--match-thresh` (`0.8`, the maximum accepted association cost — 1 − IoU, fused with detection score in the default mode, plain 1 − IoU with `--mot20`; larger accepts less-similar matches) or a longer `--track-buffer` (`60`, lost-track window scaled by `frame_rate / 30`) can help. These are tuning directions, not recalibrated thresholds. The pipeline tracks only COCO `person`; multi-class tracking needs one tracker per class or a class-aware tracker extension (see the [evaluator notes](evaluator/README.md)).

The detector keeps COCO `person` class `0`. BYTETracker first matches high-score boxes, then associates remaining tracks with low-score boxes using IoU; only unmatched high-score detections initialize new tracks.

![ByteTrack detection example strip](test_data/readme_img/image1.png)
![Three-row (a)/(b)/(c) association illustration](test_data/readme_img/image.png)

<a id="entry-points"></a>
## Entry points

- [`model/README.md`](./model/README.md): three target HBM rows and explicit preparation.
- [`runtime/python/README.md`](./runtime/python/README.md): stateful task API and full CLI.
- [`conversion/README.md`](./conversion/README.md): detector-only conversion boundary.
- [`evaluator/README.md`](./evaluator/README.md): evaluation commands, outputs, and comparison notes.

<a id="historical-performance"></a>
## Source performance reference

The source record reports tracker update time of about `2.37 ms` on RDK S100. The paper reports `80.3 MOTA`, `77.3 IDF1`, and about `30 FPS` on a V100 GPU.

<a id="license"></a>
## License

Repository helpers follow Apache-2.0. The bundled tracker source tree ships no separate license file; its provenance is recorded in `TRACKER_SOURCE_MAP.json`. Upstream ByteTrack and any model weights keep their own licenses.
