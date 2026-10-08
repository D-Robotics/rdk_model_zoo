English | [简体中文](README_cn.md)

# ByteTrack evaluator

<a id="dataset"></a>
## Dataset

The sample bundles still images and reference GIF/PNG files but no `track_test.mp4` and no MOT ground-truth directory. Prepare the video explicitly from `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4` (recorded SHA-256 `4bbe5bf11fe8967b28a900fd2add4949aba89b62076eaa03d0c55cdf7dd41397`; verify with `sha256sum track_test.mp4`, or `shasum -a 256 track_test.mp4` on macOS). The comparator checks two complete tracker captures against each other; it does not compute MOTA/IDF1 from labeled data.

<a id="directory"></a>
## Directory structure

```text
evaluator/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── capture.py  # Python script
└── compare.py  # Python script
```

<a id="environment"></a>
## Environment

A real capture requires a recognized S target board, prepared HBM/video, `hbm_runtime`, OpenCV, NumPy, SciPy, `lap==0.5.12`, and `cython-bbox==0.1.5`. `compare.py` launches each side in a fresh subprocess so process-global track IDs start consistently.

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

The tool runs `capture.py` twice: `--side unified` uses this sample's implementation, `--side legacy` runs the pinned original S-platform implementation read from the repository's Git object store — use a checkout with full history, and in a shallow clone run `git fetch origin d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d` first (the tool names this exact command when the object is missing). Each run captures every frame's image, native detector inputs/outputs, track records, metadata, model/video/code hashes, and subprocess logs under `legacy/`, `unified/`, and `comparison.json`. It returns `0` only for exact frame/input/track-ID agreement within declared box/score tolerances. Existing output directories are rejected.

<a id="metrics"></a>
## Metrics

Inputs are exact; native outputs require same shape/dtype and `rtol=0, atol=1e-5`; track IDs are exact, boxes use `atol=1e-4`, and scores `1e-5`. This is an implementation-consistency check, not MOT accuracy. Empty frames are included and must align. A non-finite track record is an error and fails the capture; it is not relaxed by `allclose`.

<a id="outputs"></a>
## Outputs

Each side retains complete `.npy` image/input/output arrays and `capture.json`; the top-level `comparison.json` retains both subprocess results and every check. Failed runs keep their error records. Fresh processes are required because track IDs are process-global; `reset` inside one process clears frame state but does not reset IDs.

<a id="reference-results"></a>
## Reference results

| Reference | Condition | Value | Source |
|---|---|---:|---|
| Source tracker update | RDK S100 | 2.37 ms average | source evaluator record |
| ByteTrack paper MOTA | MOT17 test / V100 | 80.3 | paper |
| ByteTrack paper IDF1 | MOT17 test / V100 | 77.3 | paper |
| ByteTrack paper throughput | V100 GPU | about 30 FPS | paper |

### Reference tracking effects

The source evaluator embedded two animated ByteTrack results on MOT17 `SDP` sequences as reference visualizations of the upstream method:

![MOT17-01-SDP](../test_data/readme_img/MOT17-01-SDP.gif)

`MOT17-01-SDP` sequence, reference GIF from the source evaluator (`../test_data/readme_img/MOT17-01-SDP.gif`).

![MOT17-07-SDP](../test_data/readme_img/MOT17-07-SDP.gif)

`MOT17-07-SDP` sequence, reference GIF from the source evaluator.

### Tracker parameter tuning and applicability

- `--score-thres` (default `0.25`): detector confidence filter applied before the tracker; lower it when too few boxes are detected. Lowering `--track-thresh` does not restore boxes the detector already discarded.
- `--track-thresh` (default `0.3`): partitions tracker input each frame — scores above it enter first association; scores in (0.1, track-thresh) enter second association against still-tracked targets at a fixed cost limit of `0.5`; new tracks initiate only from unmatched first-association boxes with score ≥ `track_thresh + 0.1` (`det_thresh`).
- `--match-thresh` (default `0.8`): maximum accepted cost for the first-association assignment (cost = 1 − IoU, fused with detection score in the default mode; the `--mot20` flag, default `false`, disables the fusion so the cost is plain 1 − IoU). Larger values accept less-similar matches; smaller values restrict matching to closer overlaps. The second association keeps its fixed `0.5` limit.
- `--track-buffer` (default `60`): lost-track keep window in 30 fps frames, scaled by `frame_rate / 30` (`--frame-rate`, default `30`).

For multi-class tracking, either maintain one tracker per class or extend the tracker to carry `class_id` and handle class information during association. The shipped pipeline filters to COCO `person` (class `0`) only.

<a id="boundaries"></a>
## Scope

The comparator does not download models/video and does not convert models. Both sides must use the same target HBM and video. A legacy-side letterbox zero-area/NaN record is surfaced as a failed capture; the sample runtime's filtering of non-positive person boxes is documented runtime behavior.
