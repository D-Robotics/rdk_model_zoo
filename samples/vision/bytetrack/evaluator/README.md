# ByteTrack evaluator

<a id="dataset"></a>
## Dataset

The source contains still images and reference GIF/PNG files but no `track_test.mp4` and no MOT ground-truth directory. Prepare the video explicitly from `https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ByteTrack/track_test.mp4`; this migration did not download it. The evaluator compares two complete source/unified captures, not MOTA/IDF1 from a labeled dataset.

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

The command captures every frame's image, native detector inputs/outputs, track records, metadata, model/video/code hashes, and subprocess logs under `legacy/`, `unified/`, and `comparison.json`. It returns `0` only for exact frame/input/track-ID agreement within declared box/score tolerances. Existing output directories are rejected. This migration did not run it on a board.

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

These values are historical source/paper references. Current board capture and MOT benchmark status are `not-run`.

<a id="boundaries"></a>
## Boundaries

The comparator does not download models/video, convert models, or claim board support from fake-runtime tests. Source and unified paths must use the same target HBM and video. The source letterbox zero-area/NaN edge is intentionally surfaced as failed evidence; unified filtering of non-positive person boxes is documented runtime behavior.
