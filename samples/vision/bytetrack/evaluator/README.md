# ByteTrack evaluator

<a id="dataset"></a>
## Dataset

The source contains still images and reference GIF/PNG files but no `track_test.mp4` and no MOT ground-truth directory. Prepare the video explicitly from `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4`; the 2026-09-24 board rounds downloaded exactly this video (recorded SHA `4bbe5bf1…` in the [real-video evidence](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-bytetrack-realvideo30/)), while this tree performs no new download. The evaluator compares two complete source/unified captures, not MOTA/IDF1 from a labeled dataset.

<a id="environment"></a>
## Environment

A real capture requires a recognized S target board, prepared HBM/video, `hbm_runtime`, OpenCV, NumPy, SciPy, `lap==0.5.12`, and `cython-bbox==0.1.5`. `compare.py` launches fresh subprocesses for legacy and unified capture so process-global IDs start consistently. Host tests use fake detector runtime plus real CPU tracker dependencies and are not board evidence.

<a id="command"></a>
## Evaluation command

From the repository root, after preparing model/video and choosing a new output directory:

```bash
python3 samples/vision/bytetrack/evaluator/compare.py \
  --target s100 \
  --asset-id s:bytetrack:s100/yolov5x_672x672_nv12.hbm \
  --model-path samples/vision/bytetrack/model/s100/yolov5x_672x672_nv12.hbm \
  --input samples/vision/bytetrack/test_data/track_test.mp4 \
  --output-dir /tmp/bytetrack-evidence-unique \
  --max-frames 30
```

The command captures every frame's image, native detector inputs/outputs, track records, metadata, model/video/code hashes, and subprocess logs under `legacy/`, `unified/`, and `comparison.json`. It returns `0` only for exact frame/input/track-ID agreement within declared box/score tolerances. Existing output directories are rejected. This tool ran on real boards in the 2026-09-24 records at pinned commits — the S100 four-frame synthetic case and the S100/S600 first-30-frames `track_test.mp4` cases, all checks true ([four-frame](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-bytetrack-s100/), [real-video](../../../../docs/releases/unified-migration/evidence/2026-09-24-b7-bytetrack-realvideo30/) evidence) — while the current tree adds no new board run, those records do not re-validate the current HEAD, and MOT-dataset accuracy remains `not-run`.

<a id="metrics"></a>
## Metrics

Inputs are exact; native outputs require same shape/dtype and `rtol=0, atol=1e-5`; track IDs are exact, boxes use `atol=1e-4`, and scores `1e-5`. This is a source/unified consistency check, not MOT accuracy. Empty frames are included and must align. A source non-finite track record is an error and fails the capture; it is not relaxed by `allclose`.

<a id="outputs"></a>
## Outputs

Each side retains complete `.npy` image/input/output arrays and `capture.json`; the top-level `comparison.json` retains both subprocess results and every check. Failed source/unified runs keep their error records. Fresh processes are required because track IDs are process-global; `reset()` inside one process clears frame state but does not reset IDs.

<a id="reference-results"></a>
## Reference results

| Reference | Condition | Value | status |
|---|---|---:|---|
| Source tracker update | RDK S100 | 2.37 ms average | historical/not-run |
| ByteTrack paper MOTA | MOT17 test / V100 | 80.3 | paper reference |
| ByteTrack paper IDF1 | MOT17 test / V100 | 77.3 | paper reference |
| ByteTrack paper throughput | V100 GPU | about 30 FPS | paper reference |

These values are historical source/paper references. Board source/unified captures exist for the recorded cases linked above at pinned commits; MOT benchmark (MOTA/IDF1) evaluation remains `not-run`, and this tree adds no new capture.

### Reference tracking effects (historical)

The fixed S source evaluator embedded two animated ByteTrack results on MOT17 `SDP` sequences as reference effects. They are retained verbatim as historical visualizations of the upstream method on those sequences; this migration did not rerun them on a board:

![MOT17-01-SDP](../test_data/readme_img/MOT17-01-SDP.gif)

`MOT17-01-SDP` sequence, source reference GIF (`../test_data/readme_img/MOT17-01-SDP.gif`, S pin `380e1a2`, sha256 `6b7a613f…`).

![MOT17-07-SDP](../test_data/readme_img/MOT17-07-SDP.gif)

`MOT17-07-SDP` sequence, source reference GIF (sha256 `ff99c85a…`).

### Tracker parameter tuning and applicability

Carried from the source tuning notes and verified against this sample's tracker code:

- `--score-thres` (default `0.25`): detector confidence filter applied before the tracker; lower it when too few boxes are detected. Lowering `--track-thresh` does not restore boxes the detector already discarded.
- `--track-thresh` (default `0.3`): partitions tracker input each frame — scores above it enter first association; scores in (0.1, track-thresh) enter second association against still-tracked targets at a fixed cost limit of `0.5`; new tracks initiate only from unmatched first-association boxes with score ≥ `track_thresh + 0.1` (`det_thresh`).
- `--match-thresh` (default `0.8`): maximum accepted cost for the first-association assignment (cost = 1 − IoU, fused with detection score in the default mode; the maintained `--mot20` flag, default `false`, disables the fusion so the cost is plain 1 − IoU). Larger values accept less-similar matches; smaller values restrict matching to closer overlaps. The second association keeps its fixed `0.5` limit.
- `--track-buffer` (default `60`): lost-track keep window in 30 fps frames, scaled by `frame_rate / 30` (`--frame-rate`, default `30`).

For multi-class tracking, either maintain one tracker per class or extend the tracker to carry `class_id` and handle class information during association. The shipped pipeline filters to COCO `person` (class `0`) only.

<a id="boundaries"></a>
## Boundaries

The comparator does not download models/video, convert models, or claim board support from fake-runtime tests. Source and unified paths must use the same target HBM and video. The source letterbox zero-area/NaN edge is intentionally surfaced as failed evidence; unified filtering of non-positive person boxes is documented runtime behavior.
