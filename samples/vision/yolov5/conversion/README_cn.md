# YOLOv5 转换

<a id="source-model"></a>

## 源模型

X5 源转换 Ultralytics YOLOv5 `v2.0` 与 `v7.0` 分支模型及匹配的预训练权重：

- v2.0：[分支](https://github.com/ultralytics/yolov5/tree/v2.0)，权重 `yolov5s_tag2.0.pt`
- v7.0：[分支](https://github.com/ultralytics/yolov5/tree/v7.0)，权重 `yolov5n.pt`

上游 commit 未锁定，本 sample 也不附带 exporter 脚本或 checkpoint；下文的分支/权重配对和检测头修改即源流程。

<a id="toolchain-targets"></a>

## 工具链与目标

转换在 x86 Linux 主机的 OE X5 环境内执行（`hb_mapper`、`hb_perf`、`hrt_model_exec` 由该环境提供）；板卡不是转换主机。

两个随仓 YAML 是 X5 Bayes-e 640x640 配置，保留 `O3`、latency 模式、default calibration 和 `scale_value: 0.003921568627451`。未包含 S100/S600 Nash-e 或 S100P Nash-m 的转换 YAML。不能把 X5 YAML 套到 S HBM。

<a id="export"></a>

## 导出

在上游仓库的外部 clone 中操作，不在本 sample 目录内。

### YOLOv5 tag v2.0

克隆官方仓库、切换到 `v2.0`，并下载匹配的预训练权重：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v2.0
git branch

wget https://github.com/ultralytics/yolov5/releases/download/v2.0/yolov5s.pt -O yolov5s_tag2.0.pt
python3 -m pip install -r requirements.txt
```

修改 `models/yolo.py`，使检测头输出 NHWC：

```python
def forward(self, x):
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]
```

把 `models/export.py` 复制到仓库根目录并更新默认导出参数：

```python
parser.add_argument('--weights', type=str, default='./yolov5s_tag2.0.pt', help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
```

改写 ONNX 导出块，使模型以 `opset_version=11` 导出、输出名为 `small / medium / big`、可选 `onnxsim` 简化；然后执行：

```bash
python3 export.py
```

### YOLOv5 tag v7.0

克隆、切换到 `v7.0` 并下载权重：

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
git checkout v7.0
git branch

wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
```

`models/yolo.py` 保持同样的 NHWC 检测头修改。更新 `export.py`，使其仅导出 ONNX、`opset=11`、输出名为 `small / medium / big`：

```python
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s_tag6.2.pt', help='model.pt path(s)')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
parser.add_argument('--include', nargs='+', default=['onnx'], help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs')
```

然后执行 `python3 export.py`。

<a id="calibration"></a>

## 校准

YAML 指向 `./calibration_data_rgb_f32_coco_640`，声明 `cal_data_type: float32`、`calibration_type: default`。未随附校准数据生成器；该目录与代表性 COCO tensor 是外部前置，不能把随附推理图当校准集。

<a id="compile"></a>

## 编译

在 `samples/vision/yolov5/conversion` 中，把匹配的外部 ONNX 和校准目录放到所选 YAML 指定路径后：

```bash
# v2.0
hb_mapper checker --model-type onnx --march bayes-e --model yolov5s_tag_v2.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml

# v7.0
hb_mapper checker --model-type onnx --march bayes-e --model yolov5n_tag_v7.0_detect.onnx
hb_mapper makertbin --model-type onnx --config yolov5_detect_bayese_640x640_nv12.yaml
```

预期产物是 YAML 工作目录下按前缀命名的文件，例如 `yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin`。NCHW YAML 作为源参考保留；其 runtime 输入声明是 NCHW，而发布的 X5 runtime 路径是 NV12，必须有意选择配置。

<a id="validation"></a>

## 转换后验证

在 OE 主机/板端环境可视化编译产物并检查输入输出：

```bash
hb_perf yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
hrt_model_exec model_info --model_file yolov5s_tag_v2.0_detect_640x640_bayese_nv12.bin
```

成功条件是三个 `(1,H/stride,W/stride,255)` head、640 输入 metadata 与所选目标路径的输出行为一致。`model/README_cn.md` 中的 2026-09-24 板端记录针对发布 runtime 制品，与本地转换无关。

<a id="artifacts"></a>

## 产物

`yolov5_detect_bayese_640x640_nchw.yaml` 和 `yolov5_detect_bayese_640x640_nv12.yaml` 分别定义 X5 NCHW 与 NV12 转换配置。发布 runtime 制品见 `model/README_cn.md`。

<a id="known-gaps"></a>

## 缺失项

- 没有锁定上游 commit、checkpoint、exporter、校准生成器或 S 转换配方。
- X5 Python 与 S 的物理 tensor 协议、NMS 和反量化不同，不能用一段共享转换说明替代 target-specific metadata。
- manifest 发布 SHA-256 未知。
