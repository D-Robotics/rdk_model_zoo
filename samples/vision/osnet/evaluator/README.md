English | [简体中文](README_cn.md)

# OSNet evaluation

<a id="dataset"></a>
## Dataset

Acquire [Market-1501-v15.09.15](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) under its dataset terms and extract to /data/Market-1501-v15.09.15. Expected directories: bounding_box_train (12936), query (3368), bounding_box_test (15913 after excluding pid=-1). Archive MD5 in the original record: 65005ab7d12ec1c44de4eeafe813e68a. Training images are calibration only; full evaluation uses query + gallery.

<a id="directory"></a>
## Directory

main.py captures or re-scores features; metrics.py defines the retrieval protocol; evidence/ preserves original measurements and source identities.

<a id="environment"></a>
## Environment

X5 capture uses the same runtime and preprocessing as the Python sample. Host ONNX capture additionally needs ONNX Runtime. NumPy/Pillow are needed for capture/scoring. Existing captures can be scored on the host without a board SDK. Keep about 40 MB per feature matrix plus the dataset. Runtime timings are measured separately with the C++ executable.

<a id="command"></a>
## Commands

Run at repository root. Output directories must be new. The first command executes on X5; the others run on a host. Wall time depends on the machine and includes capture/scoring, so it is not BPU latency.

```bash
python3 samples/vision/osnet/evaluator/main.py \
  --dataset /data/Market-1501-v15.09.15 \
  --model-path samples/vision/osnet/model/spatial-rgb-late8.bin \
  --output-dir /data/osnet-eval-x5
python3 samples/vision/osnet/evaluator/main.py --backend onnx \
  --dataset /data/Market-1501-v15.09.15 \
  --model-path /work/weights/osnet_x1_0_spatial.onnx --output-dir /data/osnet-eval-float
python3 samples/vision/osnet/evaluator/main.py \
  --dataset /data/Market-1501-v15.09.15 \
  --capture-dir /data/osnet-eval-x5 --output-dir /data/osnet-score-raw --raw
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| --dataset | required | Market1501 root / 数据集根目录 |
| --output-dir | required | New result directory / 新结果目录 |
| --capture-dir | null | Re-score saved raw features / 复算特征 |
| --backend | x5 | x5 or onnx / 执行后端 |
| --model-path | null | Required for capture / 采集必需 |
| --variant | spatial-rgb-late8 | Registered X5 variant / X5 登记版本 |
| --raw | false | Score without L2 / 不做 L2 检索 |

<a id="metrics"></a>
## Metrics

Sort filenames; exclude pid=-1 but retain pid=0 distractors. For each query exclude same-person/same-camera gallery images. Rank-1 is the first valid match rate; mAP averages per-query precision at matching ranks. Default distance is squared Euclidean on L2-normalized features, equivalent to cosine ordering; --raw uses unnormalized features. No reranking. Queries without valid matches are excluded from the denominator.

<a id="outputs"></a>
## Outputs

Each result directory contains raw features.npy, ordered manifest.json, and accuracy.json with rank1_percent/mAP_percent/valid_queries/distance/provenance. Re-scoring verifies the saved manifest against dataset order. ONNX and cached-feature runs record board_test=not-run; only successful X5 capture records executed.

<a id="reference-results"></a>
## Reference results

| x1.0 variant | Rank-1 | mAP | C++ E2E P50 / P95 |
| --- | --- | --- | --- |
| FP32 reference | 94.2399% | 83.5696% | — |
| spatial-rgb-late8 | 94.0618% | 83.3643% | 8.614 / 9.641 ms |
| spatial-rgb-int16 | 94.2102% | 83.5551% | 9.018 / 10.040 ms |
| spatial-rgb | 93.3195% | 81.7215% | 1.670 / 3.488 ms |

Original 2026-10-29 measurements used 3,368 query + 15,913 gallery images, L2-normalized squared Euclidean distance and no reranking. C++ used 50 warmups, 3,000 iterations, batch 1, one stream, BPU 996 MHz and CPU schedutil. E2E covers decoded 64×128 RGB crop → Pillow-equivalent NEON resize to 128×256 → synchronous runtime → L2. Decoding, detection and matching are excluded; other crop sizes cannot inherit these timings.

[Original report](evidence/2026-10-29-report.zh-CN.md) · [Evidence scope](evidence/README.md).

<a id="boundaries"></a>
## Scope

Full CLI evaluation requires the complete Market1501 query/gallery sizes and 512-dimensional features. A 32-image simulator comparison cannot replace retrieval accuracy. The original C++ benchmark is single-stream; no two-stream throughput is claimed. Rebuilding or changing crop preprocessing requires a new comparison.

### Workbench local acceptance

Use `evaluator/workbench.py measure --weights WEIGHTS --output MEASUREMENT` to count executed inference parameters and Conv/Linear MACs (2 FLOPs/MAC; classifier excluded). Commit the reviewed sample before `freeze --workspace WB --dataset DATASET --model BIN --measurement MEASUREMENT --guide GUIDE --fixture FIXTURE --build-id ID --image-id IMAGE_ID`. The immutable campaign pins the source, artifact, complete image-byte manifest and quality budget before evaluation. ReID uses 3,368 query + 15,913 valid gallery images, L2, squared Euclidean distance, same-ID/same-camera exclusion and no reranking. Local technical acceptance permits at most 0.01 ratio loss in each of Rank-1 and mAP; publication approval remains separate.

Use the native evaluator for both the spatial ONNX and reused BIN. `--onnx-threads` defaults to 4. `--capture-only` saves the complete features and image-byte manifest without scoring; score that capture on the host with `--capture-dir`, which checks `dataset-sha256.json` against the current Market1501 image bytes. Capture-only is not an accuracy pass. Preserve `capture.json` as well as the later `accuracy.json`.

```bash
python3 samples/vision/osnet/evaluator/main.py --backend onnx --model-path /absolute/spatial.onnx --dataset /absolute/Market1501 --output-dir /absolute/new/float
python3 samples/vision/osnet/evaluator/main.py --backend x5 --model-path /absolute/spatial-rgb-late8.bin --dataset /absolute/Market1501 --output-dir /absolute/new/board --capture-only
python3 samples/vision/osnet/evaluator/main.py --capture-dir /absolute/board --dataset /absolute/Market1501 --output-dir /absolute/new/board-score
```

Run native `hrt_model_exec perf` with verified fixed input, concurrency 1 and 2, separate 20-frame device warmup and three 200-total-frame rounds. Save all commands and raw logs. `workbench.py aggregate-runtime --logs LOGS --input INPUT_BIN --output PERFORMANCE` checks total frame counts and computes FPS from completed frames/common wall time. Run the C++ benchmark above for 1/2 streams, preserving per-frame data and environment before/during/after. No clock/governor change is implied. Generate an OE analysis of the exact BIN with `hb_perf`, recording the new report's tool version; do not claim it is the missing original conversion report.

Keep full evidence and models outside Git. A local candidate package contains the BIN, release.json, SHA256SUMS and oe_report.html; release approval and publication remain pending until explicitly authorized. New Web preview consumes this package via the OSNet candidate adapter, with Rank-1/mAP ratios and mixed-INT8/16 precision.

After collecting board evidence under `runs/benchmark/BUILD/work/board`, use `stage_local.py --workspace WB --build-id BUILD --float-run-id FLOAT_BUILD --oe-run-id OE_BUILD`. It validates the full dataset identity, frozen quality budget, exact model/input/output bytes, six Runtime rounds, C++ per-frame stage sums and common-wall FPS before writing a local candidate package. `oe_report.html` is the original standalone hb_perf subgraph HTML; the complete original report tree remains in the conversion run. Layer similarity is unavailable when the historical conversion report was not recovered. Human publication approval is never synthesized.

S-series workflow: [OSNet S100/S100P/S600 Workbench acceptance](S_SERIES.md).
