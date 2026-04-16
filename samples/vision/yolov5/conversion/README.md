# English | [简体中文](./README_cn.md)

# YOLOv5 Conversion

This directory describes the conversion workflow of YOLOv5 for RDK X5.

---

## Conversion Assets

The following yaml files are kept as the reference conversion configurations:

- `yolov5_detect_bayese_640x640_nchw.yaml`
- `yolov5_detect_bayese_640x640_nv12.yaml`

---

## Output Protocol

Both YOLOv5 `v2.0` and `v7.0` detection models use the same X5 runtime output protocol:

- Input: `1x3x640x640`, `UINT8`, `NV12`
- Output 0: `1x80x80x255`, `FLOAT32`
- Output 1: `1x40x40x255`, `FLOAT32`
- Output 2: `1x20x20x255`, `FLOAT32`

The Python runtime in this sample uses this contract and decodes it with the YOLOv5 anchor-based logic.

---

## YOLOv5 Tag v2.0

### Environment Preparation

Clone the official YOLOv5 repository and switch to `v2.0`:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v2.0
git branch
```

Download the matching pretrained weights:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -O yolov5s_tag2.0.pt
```

### Export to ONNX

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

Replace the ONNX export block so the model is exported with:

- `opset_version=11`
- output names `small / medium / big`
- optional `onnxsim` simplify pass

Then run:

```bash
python3 export.py
```

### PTQ Conversion

```bash
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v2.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

### Validation

Visualize the compiled model:

```bash
hb_perf yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

Check model inputs and outputs:

```bash
hrt_model_exec model_info --model_file yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

---

## YOLOv5 Tag v7.0

### Environment Preparation

Clone the official YOLOv5 repository and switch to `v7.0`:

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
git branch
```

Download the matching pretrained weights:

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

### Export to ONNX

Keep the same NHWC detection head modification in `models/yolo.py`:

```python
def forward(self, x):
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
```

Update `export.py` so it exports ONNX only, uses `opset=11`, and sets output names to `small / medium / big`:

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_tag6.2.pt', help='model.pt path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
parser.add_argument('--include', nargs='+', default=['onnx'], help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```

Then run:

```bash
python3 export.py
```

### PTQ Conversion

```bash
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

### Validation

```bash
hb_perf yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
```

---

## Notes

- This document focuses on the conversion steps required for YOLOv5 on RDK X5.
- The runtime uses converted `.bin` models through `hbm_runtime`.
- Benchmark figures and additional reference assets are available under `test_data/` and related repository resources.
