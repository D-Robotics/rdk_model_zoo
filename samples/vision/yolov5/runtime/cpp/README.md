# YOLOv5 native C++ runtime

This directory is the native C++ counterpart of the unified YOLOv5 sample. It keeps the X5 HB-DNN adapter and the S UCP adapter separate while sharing only the host-testable head validation and decode code. The native binary has four responsibilities: parse already-resolved arguments, perform target-specific input/forward I/O, decode the three raw heads, and pass detections to the separate OpenCV visualizer. Model publication facts are resolved by `launcher.py` through `samples.vision.yolov5.runtime.python.model_binding`.

## Supported targets and assets

| target | default asset | other published assets | input | native output contract |
|---|---|---|---|---|
| `x5` | `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` (`n-v7.0`) | the nine X5 `n/s/m/l/x-v2.0` and `s/m/l/x-v7.0` assets | one packed NV12 tensor, 640x640 | three F32 NHWC heads, 80/40/20 with 255 channels |
| `s100` | `s100/yolov5x_672x672_nv12.hbm` (`x-672`) | none | split NV12 Y/UV tensors, 672x672 | three metadata-described heads; source S dequantization is applied after cache invalidation |
| `s600` | `s600/yolov5x_672x672_nv12.hbm` (`x-672`) | none | split NV12 Y/UV tensors, 672x672 | same S contract |
| `s100p` | none | none | — | rejected because YOLOv5 has no published S100P asset |

The external `--model-path` rule is strict: it is accepted only together with the exact `--asset-id` from the manifest. The path does not identify a model by its filename. The launcher verifies the complete publication row before a native executable is selected.

## Build and run

The CMake target is explicit and never reads sysfs during configuration. Build each adapter separately on a target with its native SDK:

```bash
cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/x5 -DYOLOV5_TARGET=x5
cmake --build samples/vision/yolov5/runtime/cpp/build/x5
cmake -S samples/vision/yolov5/runtime/cpp -B samples/vision/yolov5/runtime/cpp/build/s100 -DYOLOV5_TARGET=s100
cmake --build samples/vision/yolov5/runtime/cpp/build/s100
```

The host-side launcher is safe for `--help`, `--list-models`, and explicit-target `--dry-run`; those modes do not inspect a board or require an SDK. A real run requires the launcher’s target identity gate and a board matching the selected target:

```bash
samples/vision/yolov5/runtime/cpp/run.sh --target x5 --variant n-v7.0 --dry-run
samples/vision/yolov5/runtime/cpp/run.sh --target x5 --asset-id <exact-asset-id> --model-path /absolute/model.bin --test-img /absolute/bus.jpg
```

Parameters and defaults are: `--target auto`, `--variant` omitted (X5 `n-v7.0`, S `x-672`), `--asset-id` omitted unless `--model-path` is supplied, `--test-img` selected from sample test data, `--output result.jpg`, `--score-thres 0.25`, `--nms-thres 0.45`, `--priority 0`, and `--bpu-core -1` (runtime default). `--label-file` is optional. `--binary` is a host/testing override for an already-built executable.

## Interface and lifecycle

`yolov5::RuntimeOptions` is the native entry contract. `run_native` owns target-specific model initialization, tensor allocation, cache operations, synchronous forward, and complete cleanup. The X5 adapter requires exactly one packed model, one input, three outputs, rank-4 NHWC heads, and `3 * (5 + classes)` channels. It uses RAII-style cleanup for every input/output buffer and task on success and error paths. The S adapter requires one packed model, two inputs, and three outputs, submits through UCP with the caller’s `priority` and selected BPU core, and frees every UCP tensor and task.

The X5 C++ source uses letterbox-compatible native tensor handling in the legacy sample family; the unified Python path uses stretch by default. This difference is intentional and must be reported in comparisons. The S source also uses letterbox. The three heads are matched by actual metadata stride (8, 16, 32), never by output order or filename. Decode applies sigmoid logits, anchors, score filtering, and class-wise NMS. S output conversion uses the source `dequantizeTensorS32` path when the SDK build wires that helper; no scale is inferred from an asset name.

## Validation status

The host test verifies separation of the numeric core, exact-head uniqueness, explicit CMake target selection, launcher identity delegation, and the public parameter/documentation contract. Native SDK compilation, model execution, board identity, rendering, and native S/X5 tensor values are **not-run** in this migration because this host has no target SDK, board, or model asset. A successful host test is therefore a contract/decoder result, not a board performance or accuracy result.
