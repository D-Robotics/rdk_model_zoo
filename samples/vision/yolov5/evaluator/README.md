# YOLOv5 evaluator

<a id="dataset"></a>
## Dataset

The source supplies `test_data/bus.jpg` for X5 and `test_data/kite.jpg` for S, plus `coco_classes.names`; there is no labeled benchmark harness in this sample. The evaluator compares one complete source/unified run on the same image, target, artifact, and thresholds. It is a consistency evidence tool, not an mAP evaluator.

<a id="environment"></a>
## Environment

Run on a host controlling a recognized target board with Python, NumPy, OpenCV, and the target `hbm_runtime`; the evaluator also needs the fixed source runtime import path. The host tests inject a fake runtime and do not certify hardware. No model or image is downloaded by the evaluator.

<a id="command"></a>
## Evaluation command

From the repository root, after preparing the exact model and using a recognized board, choose a new empty evidence directory:

```bash
python3 samples/vision/yolov5/evaluator/compare.py \
  --target x5 --variant n-v7.0 \
  --asset-id x5:yolov5:yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --model-path samples/vision/yolov5/model/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin \
  --test-img samples/vision/yolov5/test_data/bus.jpg \
  --output-dir /tmp/yolov5-evidence-unique
```

The utility runs source and unified paths, stores complete native input/output/result arrays as `.npy`, records metadata, code/model/image hashes, thresholds and board identity in `comparison.json`, and returns `0` only when every declared comparison passes. The output directory must not already exist. This migration did not run it on a board.

<a id="metrics"></a>
## Metrics

Inputs are exact; raw output arrays use shape/dtype checks and `rtol=0, atol=1e-5`; result boxes use `atol=1e-4`, scores `1e-5`, and class IDs exact. X5 and S must be compared using their own source protocol; a host fake fixture is not a board result. Historical performance is listed below and is not a new measurement.

<a id="outputs"></a>
## Outputs

Each run directory contains `legacy_*` and `unified_*` `.npy` arrays plus `comparison.json`. Failed preload or mismatch runs retain an error/failed record and return nonzero; no mismatch is converted to pass. The arrays preserve all captured input/output tensors and decoded result fields.

<a id="reference-results"></a>
## Reference results

The complete source historical X5 table is retained below; it was not re-run:

| Model | Size | Params | BPU throughput | Python post-process |
|---|---|---:|---:|---:|
| YOLOv5s_v2.0 | 640x640 | 7.5 M | 106.8 FPS | 12 ms |
| YOLOv5m_v2.0 | 640x640 | 21.8 M | 45.2 FPS | 12 ms |
| YOLOv5l_v2.0 | 640x640 | 47.8 M | 21.8 FPS | 12 ms |
| YOLOv5x_v2.0 | 640x640 | 89.0 M | 12.3 FPS | 12 ms |
| YOLOv5n_v7.0 | 640x640 | 1.9 M | 277.2 FPS | 12 ms |
| YOLOv5s_v7.0 | 640x640 | 7.2 M | 124.2 FPS | 12 ms |
| YOLOv5m_v7.0 | 640x640 | 21.2 M | 48.4 FPS | 12 ms |
| YOLOv5l_v7.0 | 640x640 | 46.5 M | 23.3 FPS | 12 ms |
| YOLOv5x_v7.0 | 640x640 | 86.7 M | 13.1 FPS | 12 ms |

S100/S600 and current source/unified board comparison are `not-run`.

<a id="boundaries"></a>
## Boundaries

This evaluator does not download models, build conversion artifacts, or claim board compatibility from host tests. X5 source intentionally uses its OpenCV XYXY-to-NMSBoxes quirk while S uses class-wise XYXY NMS; cross-target equality is not a valid assertion.
