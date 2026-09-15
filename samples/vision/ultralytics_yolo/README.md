# Ultralytics YOLO: shared X5 / S sample

[简体中文](README_cn.md)

One Python entry point selects X5, S100, S100P or S600 through `--platform`. This first merge covers YOLOv8 detection, segmentation, pose and classification and retains the existing YOLOv5u/v9/v10/11/12 combinations and X5 YOLOv13. Separate YOLO26, YOLOE and YOLOv5 samples are outside this migration.

The existing layout remains: `conversion/`, `evaluator/`, `model/`, `runtime/` and `test_data/`, with host regression tests in `tests/`. Python preprocessing/decoders and export patches are shared. `conversion/mapper.py` dispatches to the separate `mapper_x5.py` (hb_mapper, raw float32 rgbchw) and `mapper_s.py` (hb_compile, normalized npy) workflows. C++ remains an X5-only reference implementation.

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolov8 --task detect --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s600 --family yolov8 --task cls --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s100 --list-models
```

Remove `--dry-run` on the target board to run inference. Missing default assets are downloaded; explicit `--model-path` files are never downloaded. A recognized filename selects its family; use `--family` for custom names. Explicit platform overrides host detection; unknown hosts require a platform.

| Contract | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Artifact | .bin, bayese | .hbm, nashe/nashm/nashp |
| NV12 binding | One packed buffer | NHWC Y + UV |
| Runtime CLI NMS | 0.70 | 0.45 |
| Classification CLI resize | letterbox (1) | stretch (0) |
| Classification config-class resize | stretch (0) | stretch (0) |
| Published classifier filename | 640x640 | 224x224 |
| YOLOv10 decoder | DFL + NMS | DFL without NMS |
| C++ | X5 reference | Not supplied |

Artifact names are not proof of binary input geometry. Runtime metadata must describe positive, even, square batch-one NV12 inputs. Flat metadata requires `--input-shape HxW`; conflicting geometry is rejected. Detection decoders require three feature levels and reg=16; pose requires 17 keypoints. No board inference or real toolchain compilation has been validated in this local merge.

Host checks need NumPy, OpenCV and SciPy. Inference additionally needs `hbm_runtime` from the board image. No silent dependency installation occurs. Help, dry-run and inventory commands do not load board runtime.

Old platform Python/conversion/evaluation/download commands forward here. They require a complete repository checkout. Existing platform README tables and manifests remain Benchmark evidence; other samples still live in their platform trees. Historical tags, archive URLs and dashboard snapshots are unchanged.

[Python](runtime/python/README.md) · [Models](model/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md) · [C++](runtime/cpp/README.md)

Host tests: `python -m unittest discover -s samples/vision/ultralytics_yolo/tests`. Build catalog assets first with `npm --prefix tools/catalog-publisher run build` for the asset-inventory comparison.
