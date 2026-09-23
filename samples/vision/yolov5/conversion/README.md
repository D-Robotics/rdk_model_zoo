# YOLOv5 conversion

<a id="source-model"></a>
## Source model

The fixed X5 source describes Ultralytics YOLOv5 `v2.0` and `v7.0` branches and matching pretrained weights. The source links are [v2.0](https://github.com/ultralytics/yolov5/tree/v2.0) and [v7.0](https://github.com/ultralytics/yolov5/tree/v7.0). The source does not pin a commit or ship a complete export repository; the branch/weight pairing and required detection-head edit below are source instructions, not a conversion run performed here.

<a id="toolchain-targets"></a>
## Toolchain and targets

The two checked-in YAMLs are X5 Bayes-e configurations for 640x640. They preserve `O3`, latency mode, default calibration, and `scale_value: 0.003921568627451`. S100/S600 Nash-e and S100P Nash-m conversion YAMLs are absent from the fixed source; the S conversion README is explicitly incomplete. Do not apply the X5 YAML to an S HBM.

<a id="export"></a>
## Export

For v2.0, the source asks for the `models/yolo.py` head to emit NHWC, output names `small/medium/big`, ONNX opset 11, and an optional simplifier. It pairs a v2.0 checkout with `yolov5s_tag2.0.pt`. For v7.0 it asks for the same NHWC head edit, ONNX-only export, opset 11, and the `yolov5n.pt`/tag-v7 pairing. The repository contains no exporter script or checkpoint; the following source procedure is conditional and was not executed:

```bash
# cwd: an external clone, not this sample directory
python3 -m pip install -r requirements.txt
# edit models/yolo.py as described above; prepare the matching v2.0 or v7.0 checkpoint
python3 export.py --weights yolov5n.pt --img-size 640 640 --opset 11 --include onnx
# output: an external ONNX graph with small/medium/big heads
```

<a id="calibration"></a>
## Calibration

The verbatim YAMLs point to `./calibration_data_rgb_f32_coco_640` and `cal_data_type: float32`, `calibration_type: default`. The source provides no calibration-data generator in this sample; that directory and representative COCO tensors are external prerequisites. Do not call the bundled inference images a calibration set.

<a id="compile"></a>
## Compile

From `samples/vision/yolov5/conversion`, after placing a matching external ONNX and calibration directory at the paths named by the selected YAML:

```bash
hb_mapper checker --model-type onnx --march bayes-e --model ./onnx/yolov5n_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config ./yolov5_detect_bayese_640x640_nv12.yaml
```

The expected artifact is the YAML prefix `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin` under the YAML working directory. These commands require OE and were not run here. The NCHW YAML is retained as a source reference; its runtime input declaration is NCHW while the published X5 artifact/runtime path is NV12, so select the config intentionally.

<a id="validation"></a>
## Post-conversion validation

In the OE host/board environment, inspect the generated model and throughput:

```bash
hb_perf ./yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file ./yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
```

Success requires three `(1,H/stride,W/stride,255)` heads, the expected 640 input metadata, and output behavior matching the selected target path. No conversion or board validation was performed.

<a id="artifacts"></a>
## Artifacts

`yolov5_detect_bayese_640x640_nchw.yaml` and `yolov5_detect_bayese_640x640_nv12.yaml` are copied byte-for-byte from `platforms/x5/samples/vision/yolov5/conversion/`; they are source configuration references, not proof of a local build. Published runtime artifacts are listed in `model/README.md`.

<a id="known-gaps"></a>
## Known gaps

- No pinned upstream commit, checkpoint files, export script, calibration producer, or S conversion recipe is included.
- The source X5 Python path and S path have different physical tensor protocols and NMS/dequant behavior; a shared conversion paragraph cannot replace target-specific metadata.
- All compiler/export/calibration/board results are `not-run`; manifest publisher SHA-256 values are unknown.
