# 简体中文 | [English](./README.md)

# YOLOv5 转换说明

本目录说明 YOLOv5 在 RDK X5 上的转换流程。

---

## 转换资源

当前保留的参考 yaml 如下：

- `yolov5_detect_bayese_640x640_nchw.yaml`
- `yolov5_detect_bayese_640x640_nv12.yaml`

---

## 输出协议

YOLOv5 `v2.0` 与 `v7.0` 两条检测模型在 X5 上使用相同的输出协议：

- 输入：`1x3x640x640`，`UINT8`，`NV12`
- 输出 0：`1x80x80x255`，`FLOAT32`
- 输出 1：`1x40x40x255`，`FLOAT32`
- 输出 2：`1x20x20x255`，`FLOAT32`

本 sample 的 Python runtime 使用这一协议，并采用 YOLOv5 的 anchor-based 后处理逻辑。

---

## YOLOv5 Tag v2.0

### 环境准备

克隆官方 YOLOv5 仓库并切换到 `v2.0`：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v2.0
git branch
```

下载匹配的预训练权重：

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -O yolov5s_tag2.0.pt
```

### 导出 ONNX

修改 `models/yolo.py`，确保检测头输出为 NHWC：

```python
def forward(self, x):
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
```

将 `models/export.py` 复制到仓库根目录，并修改默认导出参数：

```python
parser.add_argument('--weights', type=str, default='./yolov5s_tag2.0.pt', help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
```

导出时建议：

- `opset_version=11`
- 输出名固定为 `small / medium / big`
- 如有需要可增加 `onnxsim` 简化步骤

执行导出：

```bash
python3 export.py
```

### PTQ 量化转换

```bash
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v2.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

### 验证命令

可视化 bin 模型：

```bash
hb_perf yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

查看模型输入输出：

```bash
hrt_model_exec model_info --model_file yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

---

## YOLOv5 Tag v7.0

### 环境准备

克隆官方 YOLOv5 仓库并切换到 `v7.0`：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
git branch
```

下载匹配的预训练权重：

```bash
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

### 导出 ONNX

同样需要保持 `models/yolo.py` 中检测头输出为 NHWC：

```python
def forward(self, x):
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
```

修改 `export.py` 中与导出相关的参数，使其只导出 ONNX，使用 `opset=11`，并固定输出名为 `small / medium / big`：

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_tag6.2.pt', help='model.pt path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
parser.add_argument('--include', nargs='+', default=['onnx'], help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```

执行导出：

```bash
python3 export.py
```

### PTQ 量化转换

```bash
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

### 验证命令

```bash
hb_perf yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin
```

---

## 说明

- 本文档聚焦 YOLOv5 在 RDK X5 上的转换步骤。
- 运行时直接使用转换后的 `.bin` 模型，并通过 `hbm_runtime` 在 RDK X5 上执行。
- benchmark 图片和其他参考资料可在 `test_data/` 与相关仓库资源中查看。
