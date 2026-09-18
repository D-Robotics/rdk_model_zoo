# Ultralytics YOLO: shared X5 / S sample

[简体中文](README_cn.md)

One Python entry point selects X5, S100, S100P or S600 through `--platform`. This first merge covers YOLOv8 detection, segmentation, pose and classification and retains the existing YOLOv5u/v9/v10/11/12 combinations and X5 YOLOv13. YOLO26 now shares this entry for detect, cls, seg, pose and obb. YOLOE, standalone YOLOv5 and yolo26_depth remain outside this migration.

The existing layout remains: `conversion/`, `evaluator/`, `model/`, `runtime/` and `test_data/`, with host regression tests in `tests/`. Python preprocessing/decoders, export selection, calibration preparation and compiler planning are shared. `conversion/workflow.py` owns the common conversion steps; `mapper_x5.py` and `mapper_s.py` retain only the X5/S compiler protocol adapters (`hb_mapper` with raw float32 `rgbchw` versus `hb_compile` with normalized `npy`). C++ remains an X5-only reference implementation.

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolov8 --task detect --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s600 --family yolov8 --task cls --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s100 --list-models
```

Remove `--dry-run` on the target board to run inference. This existing compatibility entry still downloads missing default assets; explicit `--model-path` files are never downloaded. A recognized filename selects its family; use `--family` for custom names. `--target` is an alias for `--platform`, with `auto` available for detection. Explicit selection works on a development host for listing, dry-run and `--download`; inference rejects unknown local hardware and target mismatches before downloading or loading a model. Target identity comes from [the shared registry](../../../platforms/registry.json), independently of artifact support and validation status.

| Contract | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Artifact | .bin, bayese | .hbm, nashe/nashm/nashp |
| NV12 binding | One packed buffer | NHWC Y + UV |
| Runtime CLI NMS | 0.70 | 0.45 |
| Classification CLI resize | YOLO26 stretch (0); other families letterbox (1) | stretch (0) |
| Classification config-class resize | stretch (0) | stretch (0) |
| Published classifier filename | YOLO26 224x224; older families 640x640 | 224x224 |
| YOLOv10 decoder | DFL + NMS | DFL without NMS |
| C++ | X5 reference | Not supplied |

Artifact names are not proof of binary input geometry. Runtime metadata must describe positive, even, square batch-one NV12 inputs. Flat metadata requires `--input-shape HxW`; conflicting geometry is rejected. DFL detection decoders require three feature levels and reg=16; YOLO26 uses direct four-channel LTRB at strides 8/16/32; pose requires 17 keypoints.

YOLOv8n DFL and YOLO26n direct-LTRB detection passed fixed-input old/new comparisons on X5 8GB/4GB, S100, S100P and S600. The detection tasks accept injected runner callables; see [the contract](DETECTION_CONTRACT.md). This does not validate other scales/tasks, full-dataset accuracy, performance or real toolchain compilation.

Host checks need NumPy, OpenCV and SciPy. Inference additionally needs `hbm_runtime` from the board image. No silent dependency installation occurs. Help, dry-run and inventory commands do not load board runtime.

Model preparation and listings additionally use PyYAML to read the existing
platform Manifest. `--list-models` prints exact `group:sample:filename`
references accepted by `--asset-id`; these qualify existing identities and are
not claims of board verification. For example:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --target x5 --task detect --asset-id x5:ultralytics_yolo:yolov8n_detect_bayese_640x640_nv12.bin --dry-run
```

Addresses and publisher hashes come from that Manifest. Explicit preparation
uses temporary files and hash checking where a publisher hash exists. A local
observed hash does not replace a missing publisher hash.

Old platform Python/conversion/evaluation/download commands forward here. They require a complete repository checkout. Existing platform README tables and manifests remain Benchmark evidence; other samples still live in their platform trees. Historical tags, archive URLs and dashboard snapshots are unchanged.

[Python](runtime/python/README.md) · [Models](model/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md) · [C++](runtime/cpp/README.md)

Host tests: `python -m unittest discover -s samples/vision/ultralytics_yolo/tests`. Asset-inventory checks read the X5 and S platform manifests directly.

## YOLO26: one family in this sample

Use `--family yolo26` with `--task detect|cls|seg|pose|obb` and `--platform x5|s100|s100p|s600`. Each platform has 25 published assets (five tasks × n/s/m/l/x); all classifier assets use 224×224, other tasks 640×640. This is consolidation of existing support, not 100 newly released models.

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform s600 --family yolo26 --task detect --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py --platform x5 --family yolo26 --task cls --dry-run
```

`yolo_dispatch.py` chooses the task's tensor protocol; YOLO26 direct LTRB is distinct from YOLOv8 DFL. Platform input binding, classification, drawing, downloads and evaluation infrastructure are shared. The five export patches live under `conversion/yolo26/`; only compiler pipelines remain separated by X5/S. Future families can reuse a compatible task protocol after its tensor contract is verified.

Old `platforms/{x5,s}/samples/vision/ultralytics_yolo26` paths forward here. Legacy Python adapters retain pose/segmentation return formats. X5 OBB angle offset, class-aware NMS and clipping remain distinct from S. Output binding now follows geometry instead of compiler enumeration; X5 segmentation reverses letterbox using actual input geometry instead of the previous hard-coded 640 mask scaling. These corrections require board accuracy revalidation. Historical Benchmark tables are preserved and are not measurements of this new code.
