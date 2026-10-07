# YOLOv5 conversion

<a id="source-model"></a>

## Source model

The X5 source converts Ultralytics YOLOv5 `v2.0` and `v7.0` branch models with matching pretrained weights:

- v2.0: [branch](https://github.com/ultralytics/yolov5/tree/v2.0), weights `yolov5s_tag2.0.pt`
- v7.0: [branch](https://github.com/ultralytics/yolov5/tree/v7.0), weights `yolov5n.pt`

No upstream commit is pinned and no exporter script or checkpoint ships with this sample; the branch/weight pairing and the detection-head edit below are the source procedure.

<a id="toolchain-targets"></a>

## Toolchain and targets

Conversion runs on an x86 Linux host inside the OE X5 environment (the same environment supplies `hb_mapper`, `hb_perf`, `hrt_model_exec`); the board is not a conversion host.

The two checked-in YAMLs are X5 Bayes-e configurations for 640x640. They preserve `O3`, latency mode, default calibration, and `scale_value: 0.003921568627451`. S100/S600 Nash-e and S100P Nash-m conversion YAMLs are not included. Do not apply the X5 YAML to an S HBM.

<a id="export"></a>

## Export

Work in an external clone of the upstream repository, not in this sample directory.

### YOLOv5 tag v2.0

Clone the official repository, switch to `v2.0`, and download the matching pretrained weights:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v2.0
git branch

wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -O yolov5s_tag2.0.pt
python3 -m pip install -r requirements.txt
```

Modify `models/yolo.py` so the detection head exports NHWC tensors:

```python
def forward(self, x):
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
```

Copy `models/export.py` to the repository root and update the default export arguments:

```python
parser.add_argument('--weights', type=str, default='./yolov5s_tag2.0.pt', help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
```

Replace the ONNX export block so the model is exported with `opset_version=11`, output names `small / medium / big`, and an optional `onnxsim` simplify pass; then run:

```bash
python3 export.py
```

### YOLOv5 tag v7.0

Clone, switch to `v7.0`, and download the weights:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
git branch

wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

Keep the same NHWC detection-head modification in `models/yolo.py`. Update `export.py` so it exports ONNX only, uses `opset=11`, and sets output names to `small / medium / big`:

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_tag6.2.pt', help='model.pt path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
parser.add_argument('--include', nargs='+', default=['onnx'], help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```

Then run `python3 export.py`.

<a id="calibration"></a>

## Calibration

The YAMLs point to `./calibration_data_rgb_f32_coco_640` with `cal_data_type: float32`, `calibration_type: default`. No calibration-data generator is included; that directory of representative COCO tensors is an external prerequisite. Do not call the bundled inference images a calibration set.

<a id="compile"></a>

## Compile

From `samples/vision/yolov5/conversion`, after placing a matching external ONNX and the calibration directory at the paths named by the selected YAML:

```bash
# v2.0
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v2.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml

# v7.0
hb_mapper checker --model-type onnx --march bayes-e --model yolov5n_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

The expected artifacts are the YAML prefix under the YAML working directory, e.g. `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin`. The NCHW YAML is retained as a source reference; its runtime input declaration is NCHW while the published X5 artifact/runtime path is NV12, so select the config intentionally.

<a id="validation"></a>

## Post-conversion validation

In the OE host/board environment, visualize the compiled model and check its inputs and outputs:

```bash
hb_perf yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

Success requires three `(1,H/stride,W/stride,255)` heads, the expected 640 input metadata, and output behavior matching the selected target path. The 2026-09-24 board records in `model/README.md` concern the published runtime artifacts, not a local conversion.

<a id="artifacts"></a>

## Artifacts

`yolov5_detect_bayese_640x640_nchw.yaml` and `yolov5_detect_bayese_640x640_nv12.yaml` define the X5 NCHW and NV12 conversion profiles. Published runtime artifacts are listed in `model/README.md`.

<a id="known-gaps"></a>

## Additional preparation

- No pinned upstream commit, checkpoint files, export script, calibration producer, or S conversion recipe is included.
- The source X5 Python path and S path have different physical tensor protocols and NMS/dequant behavior; a shared conversion paragraph cannot replace target-specific metadata.
- Manifest publisher SHA-256 values are unknown.
